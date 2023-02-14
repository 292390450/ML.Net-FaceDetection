using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using static System.Formats.Asn1.AsnWriter;
using static System.Net.Mime.MediaTypeNames;

namespace UltraFaceTest
{
    internal class Program
    {
        const int RequiredWidth = 640;
        const int RequiredHeight = 480;
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
           
            //加载模型
            var options = SessionOptionsContainer.Create("Platform");
            var _session = new InferenceSession(Path.Combine(AppContext.BaseDirectory, "ultraface.onnx"),options);
            //准备参数
            var stop = Stopwatch.StartNew();
            Console.WriteLine($"开始处理流程：{stop.ElapsedMilliseconds}ms");
            using (var sourceImage= SKBitmap.Decode(Path.Combine(AppContext.BaseDirectory, "1.jpg")))
            {
                Console.WriteLine($"原始图像加载完成：{stop.ElapsedMilliseconds}ms");
                var inputImage= sourceImage.Resize(new SKSizeI(RequiredWidth, RequiredHeight), SKFilterQuality.Medium);
                Console.WriteLine($"图像缩放结束：{stop.ElapsedMilliseconds}ms");
                //输入图像像素为rgba  ，模型需要rgb 且数据连续 ，处理一下
                //Input tensor is 1 x 3 x height x width with mean values 127, 127, 127 and scale factor 1.0 / 128
                float[] channelData = new float[RequiredWidth*RequiredHeight*3];
                var bytes = inputImage.GetPixelSpan();
                Console.WriteLine($"获取像素结束：{stop.ElapsedMilliseconds}ms");
                var expectedChannelLength = channelData.Length / 3;
                var greenOffset = expectedChannelLength;
                var blueOffset = expectedChannelLength * 2;
                Console.WriteLine($"开始像素处理：{stop.ElapsedMilliseconds}ms");
                for (int i = 0, i2 = 0; i < bytes.Length; i += 4, i2++)
                {
                    var r = (Single)(bytes[i]);
                    var g = (Single)(bytes[i + 1]);
                    var b = (Single)(bytes[i + 2]);
                    channelData[i2] = (r - 127.0f) / 128.0f;
                    channelData[i2 + greenOffset] = (g - 127.0f) / 128.0f;
                    channelData[i2 + blueOffset] = (b - 127.0f) / 128.0f;
                }
                Console.WriteLine($"像素处理结束：{stop.ElapsedMilliseconds}ms");
                //入参，参数请查看模型说明
                var input=  new DenseTensor<float>(new Memory<float>(channelData), new[] { 1, 3, RequiredHeight, RequiredWidth });
                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", input) };
                //执行
                Console.WriteLine($"开始预测：{stop.ElapsedMilliseconds}ms");
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
                Console.WriteLine($"预测结束：{stop.ElapsedMilliseconds}ms");
                var resultsArray = results.ToArray();
                //信度值，返回值也可查看模型，两个一组，第二个表示人脸值 
                float[] confidences = resultsArray[0].AsEnumerable<float>().ToArray();
                var scores = confidences.Where((val, index) => index % 2 == 1).ToList();
                //边框 
                float[] boxes = resultsArray[1].AsEnumerable<float>().ToArray();
                List<Tuple<Single,SKRect> > recs=new List<Tuple<Single,SKRect>>();
                int index = 0;
                Console.WriteLine($"开始后处理：{stop.ElapsedMilliseconds}ms");
                foreach (var f in scores)
                {
                    if (f > 0.95f)   //可信值超过95%的
                    {
                        var boxOffset = index * 4;

                        SKRect skRect = new SKRect(boxes[boxOffset]*sourceImage.Width, boxes[boxOffset + 1]*sourceImage.Height, boxes[boxOffset + 2]*sourceImage.Width,
                            boxes[boxOffset + 3] * sourceImage.Height);
                        var sam=  recs.FirstOrDefault(x => x.Item2.IntersectsWith(skRect));
                        if (sam!=null)
                        {
                            //谁可信值大留谁
                            if (sam.Item1<f)
                            {
                                recs.Remove(sam);
                                recs.Add(new Tuple<float, SKRect>(f, skRect));
                            }
                        }
                        else
                        {
                            recs.Add(new Tuple<float, SKRect>(f,skRect));
                        }
                    }
                    index++;
                }
                Console.WriteLine($"后处理结束：{stop.ElapsedMilliseconds}ms");
               // Console.WriteLine($"总处理时间：{stop.ElapsedMilliseconds}ms");
                //画出来
                using SKSurface surface = SKSurface.Create(new SKImageInfo(sourceImage.Width, sourceImage.Height));
                using SKCanvas canvas = surface.Canvas;
                canvas.DrawBitmap(sourceImage,0,0);
                using SKPaint textPaint = new SKPaint { TextSize = 32, Color = SKColors.White };
                using SKPaint rectPaint = new SKPaint { StrokeWidth = 2, IsStroke = true, Color = SKColors.Brown };
                foreach (var tuple in recs)
                {
                    var text = $"{tuple.Item1:0.00}";
                    var textBounds = new SKRect();
                    textPaint.MeasureText(text, ref textBounds);
                    canvas.DrawRect(tuple.Item2, rectPaint);
                    canvas.DrawText(text, tuple.Item2.Left, tuple.Item2.Top - textBounds.Height, textPaint);
                }
                canvas.Flush();
                using var snapshot = surface.Snapshot();
                using var imageData = snapshot.Encode(SKEncodedImageFormat.Jpeg, 100);
                using (var fs=File.Create(Path.Combine(AppContext.BaseDirectory,"temp.jpeg")))
                {
                    imageData.SaveTo(fs);
                }
               
            }
        }
    }
}