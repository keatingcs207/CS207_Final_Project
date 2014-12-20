/** @file collision_parallel.hpp
 *  @brief Detect collision between two objects with parallel algorithm.
 *  @author: Zhijie Zhou, Danyang Chen
 *  @usage:
 *  Include this file "collision_parallel.hpp"
 *  To compile this file, make the following changes in you Makefile:
 *      # Define the C++ compiler to use
 *      CXX := $(shell which g++) -std=c++11
 *      # Define CXX compile flags
 *      CXXFLAGS += -O3 -funroll-loops -fopenmp -W -Wall -Wextra #-Wfatal-errors
 *  
 *  Parallel functions to detect collision:
 *    template<typename TRIANGLE_ITER, typename EDGE_ITER>
 *    bool collide_triangles_edge_parallel(TRIANGLE_ITER tri_begin, TRIANGLE_ITER tri_end, 
 *                                EDGE_ITER edge_begin, EDGE_ITER edge_end) 
 *   
 *    template<typename TRIANGLE_ITER1, typename TRIANGLE_ITER2>
 *    bool collide_triangles_parallel(TRIANGLE_ITER1 begin1, TRIANGLE_ITER1 end1, 
 *                           TRIANGLE_ITER2 begin2, TRIANGLE_ITER2 end2)
 */

#include "collision.hpp"
#include "omp.h"
#include <parallel/algorithm>

/**********************************************************
 * Parallel function to detect collision between edges and triangles
 */

// function object used in collide_triangles_edge_parallel()
template<typename EDGE, typename TRIANGLE>
struct triangle_edge_collision_detector{
  EDGE e_;
  
  triangle_edge_collision_detector(EDGE e):e_(e){}

  bool operator()(TRIANGLE t){
    return intersect_EdgeTriangle(t, e_);
  }
};

template<typename TRIANGLE_ITER, typename EDGE_ITER>
bool collide_triangles_edge_parallel(TRIANGLE_ITER tri_begin, TRIANGLE_ITER tri_end, 
                            EDGE_ITER edge_begin, EDGE_ITER edge_end) {
    
    omp_set_num_threads(20);

    for (auto e1 = edge_begin; e1 != edge_end; ++e1) {
        // parallel algorithm to find collision
        auto find_it = __gnu_parallel::find_if(tri_begin, tri_end, 
                                     triangle_edge_collision_detector<typename EDGE_ITER::value_type,
                                     typename TRIANGLE_ITER::value_type>(*e1));
        if (find_it != tri_end){
          return true;
        }
    }
    return false;
}


/**********************************************************
 * Parallel function to detect collision between triangles
 */

// function object used in collide_triangles_parallel()
template<typename TRIANGLE>
struct triangles_collision_detector{
  TRIANGLE t_;
  
  triangles_collision_detector(TRIANGLE t):t_(t){}

  bool operator()(TRIANGLE t2){
    return collide_triangles(t_, t2);
  }
};

template<typename TRIANGLE_ITER1, typename TRIANGLE_ITER2>
bool collide_triangles_parallel(TRIANGLE_ITER1 begin1, TRIANGLE_ITER1 end1, 
                       TRIANGLE_ITER2 begin2, TRIANGLE_ITER2 end2) {
 
    omp_set_num_threads(20);

    for (auto t1 = begin1; t1 != end1; ++t1) {
        // parallel algorithm to find collision
        auto find_it = __gnu_parallel::find_if(begin2, end2, 
                                     triangles_collision_detector<typename TRIANGLE_ITER2::value_type>(*t1));
        if (find_it != end2){
          return true;
        }
    }
    return false;
}


