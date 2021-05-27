using System;
using System.Collections.Generic;
using System.Text;

namespace OSM_Parser {
  public class Way {
    public List<ulong> Nodes = new List<ulong>();
    public bool Area = false;
  }
}
