using System.Collections.Generic;

namespace OSM_Parser {
  public class Relation {
    public List<ulong> Nodes = new List<ulong>();
    public List<ulong> InnerWays = new List<ulong>();
    public List<ulong> OuterWays = new List<ulong>();
    public List<ulong> Relations = new List<ulong>();
  }
}
