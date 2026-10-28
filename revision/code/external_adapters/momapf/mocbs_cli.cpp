/*******************************************
 * Adapter only. The MO-CBS implementation remains an external dependency.
 *******************************************/

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "api.hpp"

namespace {
std::unordered_set<std::string> closed_edges;

std::string EdgeKey(long u, long v) {
  if (u > v) std::swap(u, v);
  return std::to_string(u) + ":" + std::to_string(v);
}

void Fail(const std::string& message) {
  std::cerr << "mocbs_cli: " << message << std::endl;
  std::exit(2);
}

}  // namespace

// Called only by the temporary graph.cpp build overlay. The upstream
// MO-CBS source and its solver logic remain unchanged on disk.
extern "C" bool PNSGA3EdgeAllowed(long u, long v) {
  return closed_edges.count(EdgeKey(u, v)) == 0;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    Fail("usage: mocbs_cli INSTANCE TIME_LIMIT_SECONDS");
  }
  std::ifstream input(argv[1]);
  if (!input) Fail("cannot open instance");
  std::string version;
  input >> version;
  if (version != "PNSGA3_MOMAPF_V1" &&
      version != "PNSGA3_MOMAPF_V2") Fail("unsupported instance version");

  int rows = 0, columns = 0, agents = 0;
  input >> rows >> columns >> agents;
  if (rows <= 0 || columns <= 0 || agents <= 0) Fail("invalid dimensions");
  std::vector<std::string> map(rows);
  for (int y = 0; y < rows; ++y) {
    input >> map[y];
    if (static_cast<int>(map[y].size()) != columns) Fail("invalid map row");
  }

  rzq::basic::Grid world;
  rzq::basic::Grid travel;
  rzq::basic::Grid risk;
  world.Resize(rows, columns, 0);
  travel.Resize(rows, columns, 1);
  risk.Resize(rows, columns, 1);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < columns; ++x) {
      if (map[y][x] == '#') {
        world.Set(y, x, 1);
        continue;
      }
      int blocked = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          int ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= rows || nx < 0 || nx >= columns ||
              map[ny][nx] == '#') {
            ++blocked;
          }
        }
      }
      risk.Set(y, x, 1 + blocked);
    }
  }

  std::vector<long> starts, goals;
  for (int agent = 0; agent < agents; ++agent) {
    int sx = 0, sy = 0, gx = 0, gy = 0;
    input >> sx >> sy >> gx >> gy;
    if (!input) Fail("missing task row");
    starts.push_back(sy * columns + sx);
    goals.push_back(gy * columns + gx);
  }

  rzq::basic::GridkConn graph;
  graph.Init(world, std::vector<rzq::basic::Grid>{travel, risk});
  if (version == "PNSGA3_MOMAPF_V2") {
    int count = 0;
    input >> count;
    if (!input || count < 0) Fail("invalid closed-edge count");
    for (int i = 0; i < count; ++i) {
      int x1, y1, x2, y2;
      input >> x1 >> y1 >> x2 >> y2;
      if (!input || x1 < 0 || x1 >= columns || x2 < 0 || x2 >= columns ||
          y1 < 0 || y1 >= rows || y2 < 0 || y2 >= rows ||
          map[y1][x1] == '#' || map[y2][x2] == '#' ||
          std::abs(x1 - x2) + std::abs(y1 - y2) != 1) {
        Fail("invalid closed edge");
      }
      closed_edges.insert(EdgeKey(y1 * columns + x1, y2 * columns + x2));
    }
  }
  rzq::mapf::MOMAPFResult result;
  std::vector<long> wait_cost;
  const double time_limit = std::stod(argv[2]);
  rzq::mapf::RunMOCBS(
      graph, starts, goals, &result, time_limit, wait_cost, 1);

  std::vector<long> solution_ids;
  for (const auto& item : result.solus) solution_ids.push_back(item.first);
  std::sort(solution_ids.begin(), solution_ids.end());
  std::cout << "{\"type\":\"summary\",\"find_all_pareto\":"
            << (result.find_all_pareto ? "true" : "false")
            << ",\"solution_count\":" << solution_ids.size()
            << ",\"n_generated\":" << result.n_generated
            << ",\"n_expanded\":" << result.n_expanded
            << ",\"n_init_root\":" << result.n_initRoot
            << ",\"runtime_search\":" << result.rt_search
            << ",\"runtime_low_level\":" << result.rt_searchLow
            << ",\"low_level_expansions\":" << result.n_expandLow
            << "}" << std::endl;

  for (long id : solution_ids) {
    const auto& costs = result.costs.at(id);
    const auto& paths = result.solus.at(id);
    std::cout << "{\"type\":\"solution\",\"id\":" << id
              << ",\"cost\":[" << costs.at(0) << "," << costs.at(1)
              << "],\"paths\":[";
    for (int agent = 0; agent < agents; ++agent) {
      if (agent) std::cout << ",";
      std::cout << "[";
      const auto& path = paths.at(agent);
      for (std::size_t step = 0; step < path.size(); ++step) {
        if (step) std::cout << ",";
        const long node = path[step];
        std::cout << "[" << node % columns << "," << node / columns << "]";
      }
      std::cout << "]";
    }
    std::cout << "]}" << std::endl;
  }
  return result.find_all_pareto ? 0 : 3;
}
