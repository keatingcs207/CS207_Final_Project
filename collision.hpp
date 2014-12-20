/** @file collision.hpp
 *  @brief Detect collision between two objects.
 *  @author: Zhijie Zhou, Danyang Chen
 *  @usage:
 *  include this file "collision.hpp"
 *  
 *  Functions to detect collision between triangles and edges:
 *    template<typename TRIANGLE, typename EDGE>
 *    bool collide_triangles_edge(const TRIANGLE& t, const EDGE e)
 *
 *    template<typename TRIANGLE_ITER, typename EDGE>
 *    bool collide_triangles_edge(TRIANGLE_ITER begin, TRIANGLE_ITER end, const EDGE& e)
 *  
 *    template<typename TRIANGLE_ITER, typename EDGE_ITER>
 *    bool collide_triangles_edge(TRIANGLE_ITER tri_begin, TRIANGLE_ITER tri_end, 
 *                                EDGE_ITER edge_begin, EDGE_ITER edge_end) 
 *
 *  Functions to detect collision between triangles:
 *    template<typename TRIANGLE1, typename TRIANGLE2>
 *    bool collide_triangles(const TRIANGLE1& t1, const TRIANGLE2& t2)
 *
 *    template<typename TRIANGLE_ITER1, typename TRIANGLE_ITER2>
 *    bool collide_triangles(TRIANGLE_ITER1 begin1, TRIANGLE_ITER1 end1, 
 *                           TRIANGLE_ITER2 begin2, TRIANGLE_ITER2 end2)
 *
 */

#include <iostream>
#include "Point.hpp"
#include "collision_helper.hpp"

/** Check whether the edge @a e intersects with the triangle @a t
 * @param[in] e The edge
 * @param[in] t The triangle
 * @returns true if the edge @a e intersects with the triangle @a t,
 *          false otherwise.
 *
 * @tparam TRIANGLE is a triangle object which provides the function 
 *                  node(i) to get its ith vertex.
 * @tparam EDGE is an edge object which provides the functions 
 *                  node1() and node2() to get its vertices.
 * The vertices of the edge and triangle should be node objects which
 *                  provide function position() to get their positions.
 *
 * Complexity: O(1)
 */
template<typename TRIANGLE, typename EDGE>
bool collide_triangles_edge(const TRIANGLE& t, const EDGE e) {
    return intersect_EdgeTriangle(t, e);
}


/** Check whether two triangles @a t1 and @a t2 collide
 * @param[in] t1, t2 The two triangles
 * @returns true if the two triangles collide,
 *          false otherwise.
 *
 * @tparam TRIANGLE1 is a triangle object that provides the function
 *                  node(i) to get its ith vertex.
 * @tparam TRIANGLE2 is a triangle object that provides the function
 *                  edge(i) to get its ith edge, which supports the
 *                  functions node1() and node2() to get its vertices.
 *
 * The vertices should be node objects that provide function
 *                  position() to get their positions.
 *
 * Complexity: O(1)
 */
template<typename TRIANGLE1, typename TRIANGLE2>
bool collide_triangles(const TRIANGLE1& t1, const TRIANGLE2& t2) {
    for (unsigned i = 0; i < 3; ++i){
        if (intersect_EdgeTriangle(t1, t2.edge(i))){
            return true;       // t1 and t2 collide if any edge of t2 intersects with t1
        }
    }
    return false;
}


/** Check whether any of the triangles in the range [@a begin, @a end) 
 *  collides with the edge @a e
 * 
 * @param[in] begin, end The begin and end iterators of the triangles
 * @param[in] e The edge
 * @returns true if any triangle collides with @a e,
 *          false otherwise.
 *
 * @tparam TRIANGLE_ITER is a iterator points to triangle objects.
 *                  The triangle objects should provide the function 
 *                  node(i) to get the ith vertex.
 * @tparam EDGE is an edge object which provides the functions 
 *                  node1() and node2() to get its vertices.
 * The vertices of the edge and triangle should be node objects which
 *                  provide function position() to get their positions.
 *
 * Complexity: O(number of triangles)
 */
template<typename TRIANGLE_ITER, typename EDGE>
bool collide_triangles_edge(TRIANGLE_ITER begin, TRIANGLE_ITER end, const EDGE& e){
    for (auto iter = begin; iter != end; ++iter){
        if (intersect_EdgeTriangle(*iter, e)){
            return true;
        }
    }
    return false;
}


/** Check whether any of the triangles in the range [@a tri_begin, @a tri_end) 
 *  collides with any of the edges in the range [@a edge_begin, @a edge_end)
 * 
 * @param[in] tri_begin, tri_end The begin and end iterators of the triangles
 * @param[in] edge_begin, edge_end The begin and end iterators of the edges
 * @returns true if any collision detected,
 *          false otherwise.
 *
 * @tparam TRIANGLE_ITER is a iterator points to triangle objects.
 *                  The triangle objects should provide the function 
 *                  node(i) to get the ith vertex.
 * @tparam EDGE_ITER is a iterator points to edge objects. The edge 
 *                  objects should provide the functions 
 *                  node1() and node2() to get its vertices.
 * The vertices of the edge and triangle should be node objects which
 *                  provide function position() to get their positions.
 *
 * Complexity: O(number of triangles * number of edges)
 */
template<typename TRIANGLE_ITER, typename EDGE_ITER>
bool collide_triangles_edge(TRIANGLE_ITER tri_begin, TRIANGLE_ITER tri_end, 
                            EDGE_ITER edge_begin, EDGE_ITER edge_end) {
    
    for (auto e_it = edge_begin; e_it != edge_end; ++e_it) {
        for (auto t_it = tri_begin; t_it != tri_end; ++t_it) {
            if (intersect_EdgeTriangle(*t_it, *e_it)){
                return true;       
            }
        }
    }
    return false;
}


/** Check whether any of the triangles in the range [@a begin1, @a end1) 
 *  collides with any of the triangles in the range [@a begin2, @a end2)
 * 
 * @param[in] begin1, end1, begin2, end2 The begin and end iterators of the triangles
 * @returns true if any collision detected,
 *          false otherwise.
 *
 * @tparam TRIANGLE_ITER1, TRIANGLE_ITER2 are iterators point to triangle objects.
 *                  The triangle objects should provide the function node(i) to
 *                  get the ith vertex, and edge(i) to get the ith edge. 
 *                  The edge objects should provide the functions 
 *                  node1() and node2() to get its vertices.
 *                  The vertices of the edges and triangles should be node objects 
 *                  which provide function position() to get their positions.
 *
 * Complexity: O(number of triangles in the first range * number of triangles in the second range)
 */
template<typename TRIANGLE_ITER1, typename TRIANGLE_ITER2>
bool collide_triangles(TRIANGLE_ITER1 begin1, TRIANGLE_ITER1 end1, 
                       TRIANGLE_ITER2 begin2, TRIANGLE_ITER2 end2) {
    for (auto t1 = begin1; t1 != end1; ++t1) {
        for (auto t2 = begin2; t2 != end2; ++t2) {
            if (collide_triangles(*t1, *t2)){
                return true;
            }
        }
    }
    return false;
}



