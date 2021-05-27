using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using System.Xml;

namespace OSM_Parser {
  class OSMParser {
    // Props
    public float XLowerLeftCorner { get; set; }
    public float YLowerLeftCorner { get; set; }
    public float CellSize { get; set; }

    private readonly string FileName;
    private int width;
    private int height;

    // Constructors
    public OSMParser(string fileName) {
      XLowerLeftCorner = 0;
      YLowerLeftCorner = 0;
      CellSize = 0;
      FileName = fileName;
    }
    public OSMParser(float xLoerLeftCorner, float yLowerLeftCorner, float cellSize, int width, int height, string fileName) {
      XLowerLeftCorner = xLoerLeftCorner;
      YLowerLeftCorner = yLowerLeftCorner;
      CellSize = cellSize;
      this.width = width;
      this.height = height;
      FileName = fileName;
    }

    // Parse methods
    
    public Bitmap ParseRoads() {

      Console.WriteLine("Parsing Roads . . .");

      var roads = new Bitmap(width, height);
      Graphics g = Graphics.FromImage(roads);
      g.Clear(Color.Black);
      var roadTypes = new HashSet<string> {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary"
      };

      using (XmlReader reader = XmlReader.Create(FileName)) {
        var nodes = ParseAllNodes(reader);

        foreach ((ulong _, Way way, Dictionary<string, string> tags) in ParseWays(reader))
          if (tags.TryGetValue("highway", out string val) && roadTypes.Contains(val))
            DrawWay(way, nodes, g);
      }

      g.Dispose();
      return roads;
    }

    public Bitmap ParseRivers() {
      Console.WriteLine("Parsing Rivers . . .");

      var rivers = new Bitmap(width, height);
      Graphics g = Graphics.FromImage(rivers);
      g.Clear(Color.Black);

      using (XmlReader reader = XmlReader.Create(FileName)) {
        var nodes = ParseAllNodes(reader);
        var ways = new Dictionary<ulong, Way>();

        foreach ((ulong id, Way way, Dictionary<string, string> tags) in ParseWays(reader)) {
          if (tags.TryGetValue("waterway", out string val) && (true
            ))
            DrawWay(way, nodes, g);

          ways[id] = way;
        }

        foreach ((ulong _, Relation relation, Dictionary<string, string> tags) in ParseRelations(reader)) {
          if (tags.TryGetValue("water", out string val) && val == "river")
            DrawRelation(relation, ways, nodes, g);
        }
      }

      g.Dispose();
      return rivers;
    }

    public Bitmap ParseBuildings() {
      Console.WriteLine("Parsing Buildings . . .");
      Bitmap buildings = new Bitmap(width, height);
      Graphics g = Graphics.FromImage(buildings);
      //g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.None;
      g.Clear(Color.Black);

      using (XmlReader reader = XmlReader.Create(FileName)) {
        var nodes = ParseAllNodes(reader);

        foreach ((ulong key, Way way, Dictionary<string, string> tags) in ParseWays(reader)) {
          if (tags.ContainsKey("building") || tags.ContainsKey("amenity")) {
            /**/
            if (way.Area)
              FillArea(way, nodes, g, Brushes.White);
            else
              DrawWay(way, nodes, g);
            /**/
          }
        }
      }

      g.Dispose();
      return buildings;
    }

    public IEnumerable<(ulong, Node, Dictionary<string, string>)> ParseNodes(XmlReader reader) {
      ulong id = 0;
      Dictionary<string, string> tags = null;
      Node nd = null;

      if (reader.NodeType == XmlNodeType.Element && reader.Name == "node") {
        id = ulong.Parse(reader.GetAttribute("id"));
        tags = new Dictionary<string, string>();
        nd = new Node {
          lat = float.Parse(reader.GetAttribute("lat")),
          lon = float.Parse(reader.GetAttribute("lon")),
        };
      }

      while (reader.Read()) {

        if (reader.NodeType == XmlNodeType.Element && reader.Name == "way") break;

        if (reader.NodeType == XmlNodeType.EndElement && reader.Name == "node") {
          yield return (id, nd, tags);
          nd = null;
          tags = null;
        }

        if (reader.NodeType == XmlNodeType.Element && reader.Name == "node") {
          id = ulong.Parse(reader.GetAttribute("id"));
          tags = new Dictionary<string, string>();
          nd = new Node {
            lat = float.Parse(reader.GetAttribute("lat")),
            lon = float.Parse(reader.GetAttribute("lon")),
          };

          if (reader.IsEmptyElement)
            yield return (id, nd, tags);
        }

        if (reader.NodeType == XmlNodeType.Element && reader.Name == "tag") {
          string key = reader.GetAttribute("k");
          string val = reader.GetAttribute("v");

          tags?.Add(key, val);
        }
      }

    }

    public IEnumerable<(ulong, Way, Dictionary<string, string>)> ParseWays(XmlReader reader) {
      ulong id = 0;
      Dictionary<string, string> tags = null;
      Way way = null;

      if (reader.NodeType == XmlNodeType.Element && reader.Name == "way") {
        id = ulong.Parse(reader.GetAttribute("id"));
        tags = new Dictionary<string, string>();
        way = new Way();
      }

      while (reader.Read()) {
        if (reader.NodeType == XmlNodeType.Element && reader.Name == "relation")
          break;

        if (reader.NodeType == XmlNodeType.Element && reader.Name == "way") {
          id = ulong.Parse(reader.GetAttribute("id"));
          tags = new Dictionary<string, string>();
          way = new Way();
        }

        if (reader.NodeType == XmlNodeType.Element && reader.Name == "tag")
          tags?.Add(reader.GetAttribute("k"), reader.GetAttribute("v"));

        if (reader.NodeType == XmlNodeType.Element && reader.Name == "nd")
          way?.Nodes.Add(ulong.Parse(reader.GetAttribute("ref")));

        if (reader.NodeType == XmlNodeType.EndElement && reader.Name == "way") {
          if (way.Nodes[0] == way.Nodes[way.Nodes.Count - 1])
            way.Area = true;

          yield return (id, way, tags);
          way = null;
          tags = null;
        }
      }
    }

    public IEnumerable<(ulong, Relation, Dictionary<string, string>)> ParseRelations(XmlReader reader) {
      ulong id = 0;
      Dictionary<string, string> tags = null;
      Relation relation = null;

      if (reader.NodeType == XmlNodeType.Element && reader.Name == "relation") {
        id = ulong.Parse(reader.GetAttribute("id"));
        tags = new Dictionary<string, string>();
        relation = new Relation();
      }

      while (reader.Read()) {
        if (reader.NodeType == XmlNodeType.Element && reader.Name == "relation") {
          id = ulong.Parse(reader.GetAttribute("id"));
          tags = new Dictionary<string, string>();
          relation = new Relation();
        }

        if (reader.NodeType == XmlNodeType.Element && reader.Name == "tag")
          tags?.Add(reader.GetAttribute("k"), reader.GetAttribute("v"));

        if (reader.NodeType == XmlNodeType.Element && reader.Name == "member") {
          ulong memberId = ulong.Parse(reader.GetAttribute("ref"));
          string type = reader.GetAttribute("type");
          string role = reader.GetAttribute("role");


          if (type == "node")
            relation?.Nodes.Add(memberId);

          if (type == "way" && role == "inner")
            relation?.InnerWays.Add(memberId);

          if (type == "way" && role == "outer")
            relation?.OuterWays.Add(memberId);

          if (type == "relation")
            relation?.Relations.Add(memberId);
        }


        if (reader.NodeType == XmlNodeType.EndElement && reader.Name == "relation") {
          yield return (id, relation, tags);
          relation = null;
          tags = null;
        }
      }
    }

    public Dictionary<ulong, Node> ParseAllNodes(XmlReader reader) {
      var allNodes = new Dictionary<ulong, Node>();

      var nodesAndTags = ParseNodes(reader);
      foreach ((ulong id, Node nd, _) in nodesAndTags)
        allNodes.Add(id, nd);

      return allNodes;
    }
    public Dictionary<ulong, Way> ParseAllWays(XmlReader reader) {
      var allWays = new Dictionary<ulong, Way>();

      var waysAndTags = ParseWays(reader);
      foreach ((ulong id, Way way, _) in waysAndTags)
        allWays.Add(id, way);

      return allWays;
    }

    // Draw methods

    public void FillArea(Way area, Dictionary<ulong, Node> nodes, Graphics g, Brush b) {
      Point[] points = new Point[area.Nodes.Count];
      for (int i = 0; i < area.Nodes.Count; i++) {
        Node nd = nodes[area.Nodes[i]];
        ToDescrete(nd.lon, nd.lat, out int x, out int y);

        points[i] = new Point(x, y);
      }

      g.FillPolygon(b, points);
    }

    public void DrawWay(Way way, Dictionary<ulong, Node> nodes, Graphics g) {
      for (int i = 0; i < way.Nodes.Count - 1; i++) {
        Node from = nodes[way.Nodes[i]];
        Node to = nodes[way.Nodes[i + 1]];
        ToDescrete(from.lon, from.lat, out int fromx, out int fromy);
        ToDescrete(to.lon, to.lat, out int tox, out int toy);

        g.DrawLine(Pens.White, fromx, fromy, tox, toy);
      }
    }

    public void DrawRelation(Relation relation, Dictionary<ulong, Way> ways, Dictionary<ulong, Node> nodes, Graphics g) {
      foreach (ulong key in relation.OuterWays) {
        if (ways.TryGetValue(key, out Way w) && w.Area)
          FillArea(w, nodes, g, Brushes.White);
      }

      foreach (ulong key in relation.InnerWays) {
        if (ways.TryGetValue(key, out Way w) && w.Area)
          FillArea(w, nodes, g, Brushes.Black);
      }
    }

    // Utility methods

    private void ToDescrete(float x, float y, out int xo, out int yo) {
      xo = (int)Math.Round(Math.Max(Math.Min(((x - XLowerLeftCorner) / CellSize), width - 1), 0));
      yo = (int)Math.Round(Math.Max(Math.Min(((y - YLowerLeftCorner) / CellSize), height - 1), 0));
    }

  }
}
