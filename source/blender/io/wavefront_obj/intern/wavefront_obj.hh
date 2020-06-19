/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/** \file
 * \ingroup obj
 */

#ifndef __WAVEFRONT_OBJ_HH__
#define __WAVEFRONT_OBJ_HH__

#include <stdio.h>

#include "BKE_context.h"
#include "BKE_object.h"

#include "BLI_array.hh"
#include "BLI_vector.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

namespace io {
namespace obj {

/* -Y */
#define DEFAULT_AXIS_FORWARD 4
/* Z */
#define DEFAULT_AXIS_UP 2

/**
 * Polygon stores the data of one face of the mesh.
 * f v1/vt1/vn1 v2/vt2/vn2 .. (n)
 */
struct Polygon {
  /** Total vertices in one polgon face. n above. */
  uint total_vertices_per_poly;
  /**
   * Vertex indices of this polygon. v1, v2 .. above.
   */
  std::vector<uint> vertex_index;
  /**
   * UV vertex indices of this polygon. vt1, vt2 .. above.
   */
  std::vector<uint> uv_vertex_index;
};

/**
 * Stores geometry of one mesh object to be exported.
 */
typedef struct OBJ_obmesh_to_export {
  bContext *C;
  Depsgraph *depsgraph;
  Object *object;

  /** Vertices in a mesh to export. */
  MVert *mvert;
  /** Number of vertices in a mesh to export. */
  uint tot_vertices;

  /** Polygons in a mesh to export. */
  /* TODO (ankitm): Replace vector with BLI::Vector. See D7931 */
  std::vector<Polygon> polygon_list;
  /** Number of polygons in a mesh to export. */
  uint tot_poly;

  /** UV vertex coordinates of a mesh in texture map. */
  std::vector<std::array<float, 2>> uv_coords;
  /** Number of UV vertices of a mesh in texture map. */
  uint tot_uv_vertices;

  int forward_axis;
  int up_axis;
  float scaling_factor;
} OBJ_obmesh_to_export;

typedef struct OBJ_obcurve_to_export {
  bContext *C;
  Depsgraph *depsgraph;
  Object *object;

  /** Vertices in a mesh made from a curve to export. */
  MVert *mvert;
  /** Number of vertices in a curve to export. */
  uint tot_vertices;

  /** Vertex indices of an edge of a mesh made from a curve. */
  std::vector<std::array<uint, 2>> edge_vert_indices;
  /** Number of edges in a curve to export. */
  uint tot_edges;

  int forward_axis;
  int up_axis;
  float scaling_factor;
  bool export_curves_as_nurbs;
} OBJ_obcurve_to_export;
}  // namespace obj
}  // namespace io

#endif /* __WAVEFRONT_OBJ_HH__ */