using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

namespace OSM_Parser {

  class Program {
    static int[,] ParseHeights(string fileName, out int width, out int height, out float xllCorner, out float yllCorner, out float cellSize, int scale = 1) {

      Console.WriteLine("Parsing Heights . . .");

      var reader = new StreamReader(fileName);
      string line;

      // number of columns and rows
      //line = reader.ReadLine();
      int cols = int.Parse(reader.ReadLine().Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]);
      int rows = int.Parse(reader.ReadLine().Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]);

      width = (int)(cols * scale);
      height = (int)(rows * scale);

      // left lower corner coords - latitude and longitude
      xllCorner = float.Parse(reader.ReadLine().Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]);
      yllCorner = float.Parse(reader.ReadLine().Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]);

      cellSize = float.Parse(reader.ReadLine().Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]) / scale;

      // no idea what this is
      int nodata_value = int.Parse(reader.ReadLine().Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]);

      // Now for the actual heights,
      // we want heights[0,0] to be left lower corner,
      // as that is how the file specifies it
      float MaxHeight = float.MinValue;
      float MinHeight = float.MaxValue;
      int[,] heights = new int[width, height];
      int row = height;
      line = reader.ReadLine();
      int bonusRow = 0;
      int lastRow = 0;

      for (int mapRow = height - 1; mapRow >= 0; mapRow--) {

        if (bonusRow / height > lastRow) {
          line = reader.ReadLine();
          lastRow++;
        }

        var nums = line.Trim().Split();
        int bonusCol = 0;


        for (int mapCol = 0; mapCol < width; mapCol++) {
          int x = bonusCol / width;
          float num = float.Parse(nums[x]);
          heights[mapCol, mapRow] = (int)num;
          MaxHeight = num > MaxHeight ? num : MaxHeight;
          MinHeight = num < MinHeight ? num : MinHeight;

          bonusCol += cols;
        }

        bonusRow += rows;
      }

      return heights;
    }
    static void SaveBitmapAsPGM(Bitmap bmp, string fileName) {

      Console.WriteLine($"Saving {fileName}.pgm . . .");

      var file = new StreamWriter(fileName + ".pgm");
      file.WriteLine("P2");
      file.WriteLine(bmp.Width);
      file.WriteLine(bmp.Height);
      file.WriteLine(1);

      int black = Color.Black.ToArgb();

      for (int y = 0; y < bmp.Height; y++) {
        string separator = "";
        for (int x = 0; x < bmp.Width; x++) {
          file.Write(separator);

          int c = bmp.GetPixel(x, y).ToArgb();
          if (c == black)
            file.Write(0);
          else
            file.Write(1);


          separator = " ";
        }

        file.WriteLine();
      }

      file.Close();
    }
    static void SaveMatrix(float[,] matrix, int wid, int hei, string fileName) {
      var file = new StreamWriter(fileName + ".pgm");

      float max = float.MinValue;
      for (int y = 0; y < hei; y++)
        for (int x = 0; x < wid; x++)
          if (matrix[x, y] > max) max = matrix[x, y];


      file.WriteLine("P2");
      file.WriteLine(wid);
      file.WriteLine(hei);
      file.WriteLine(max);

      for (int y = 0; y < hei; y++) {
        string separator = "";

        for (int x = 0; x < wid; x++) {
          file.Write(separator);
          file.Write(matrix[x, y]);
          separator = " ";
        }

        file.WriteLine();
      }

      file.Close();
    }
    static void SaveMatrix(int[,] matrix, int wid, int hei, string fileName) {
      var file = new StreamWriter(fileName + ".pgm");

      float max = int.MinValue;
      float min = int.MaxValue;
      for (int y = 0; y < hei; y++)
        for (int x = 0; x < wid; x++) {
          if (matrix[x, y] > max) max = matrix[x, y];
          if (matrix[x, y] < min) min = matrix[x, y];
        }

      float l = max - min;

      file.WriteLine("P2");
      file.WriteLine(wid);
      file.WriteLine(hei);
      file.WriteLine(255);

      for (int y = 0; y < hei; y++) {
        string separator = "";

        for (int x = 0; x < wid; x++) {
          file.Write(separator);
          file.Write((int)((matrix[x, y] - min) / l * 255));
          separator = " ";
        }

        file.WriteLine();
      }

      file.Close();
    }

    static void Main(string[] args) {
      string ascFile = args[0];
      int scale = 1;

      int[,] heights = ParseHeights(ascFile, out int wid, out int hei, out float xllCorner, out float yllCorner, out float cellSize, scale);
      SaveMatrix(heights, wid, hei, "heights");

      if (args.Length <= 1)
        return;


      string osmFile = args[1];
      var parser = new OSMParser(xllCorner, yllCorner, cellSize, wid, hei, osmFile);

      var roads = parser.ParseRoads();
      SaveBitmapAsPGM(roads, "roads");

      var rivers = parser.ParseRivers();
      SaveBitmapAsPGM(rivers, "rivers");

      var buildings = parser.ParseBuildings();
      SaveBitmapAsPGM(buildings, "buildings");
    }
  }
}