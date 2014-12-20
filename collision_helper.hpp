/*
 * helper functions for collision.hpp
 * reference: 
 * http://geomalgorithms.com/a06-_intersect-2.html
 * http://www.cnblogs.com/graphics/archive/2010/08/05/1793393.html
 * http://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
 */

using namespace std; 
#define SMALL_NUM   0.00000001 // avoids division overflow

/* intersect_EdgeTriangle(): find the intersection of an edge with a triangle
 * @param[in]:  an edge E, and a triangle T
 * @param[out]: false = no intersection
 *              true = intersection
 * Complexity: O(1)
 */
template<typename TRIANGLE, typename EDGE>
bool intersect_EdgeTriangle(const TRIANGLE& T, const EDGE& E){
    Point    u, v, n;              // triangle vectors
    Point    dir, w0, w;           // edge vectors
    double   r, a, b;              // params to calculate edge-triangle intersect
    Point    I;                    // intersection point (when it exists)

    // get triangle edge vectors and plane normal
    auto t0_pos = T.node(0).position();
    auto t1_pos = T.node(1).position();
    auto t2_pos = T.node(2).position();
    u = t1_pos - t0_pos;
    v = t2_pos - t0_pos;
    n = cross(u, v);               
    if (n == Point(0, 0, 0))       // triangle is degenerate
        return false;              // invalid case

    auto e0_pos = E.node1().position();
    auto e1_pos = E.node2().position();
    dir = e1_pos - e0_pos;         // edge direction vector
    w0 = e0_pos - t0_pos;
    a = -dot(n, w0);
    b = dot(n,dir);
    if (fabs(b) < SMALL_NUM) {     // edge is  parallel to triangle plane
        if (a == 0) {              // edge lies in the triangle plane
            if (PointinTriangle(T, e0_pos) || PointinTriangle(T, e1_pos)) {
                return true;       //edge is parallel to the triangle and one of the nodes is 
        	                       //inside of the triangle
            } 
            else if (doIntersect(e0_pos, e1_pos, t0_pos, t1_pos) || // Edge intersects any of the
                     doIntersect(e0_pos, e1_pos, t0_pos, t2_pos) || // three sides of the tringle
                     doIntersect(e0_pos, e1_pos, t1_pos, t2_pos) ) {
                return true;
            }

            else {                
                return false;      // edge is lies in the triangle plane, edge does not intersect triangle          
            }    
        }
        else return false;         // edge disjoint from plane
    }

    // get intersect point of ray with triangle plane
    r = a / b;
    if (r < 0.0 || r > 1.0)        // edge goes away from triangle
        return false;              // no intersect

    I = e0_pos + r * dir;          // intersect point of edge and plane

    return PointinTriangle(T, I);  // return true if I is inside T
}


/* Check if a point P is inside a triangle T
 * @param[in] P A point we are going to check
 * @param[in] T A triangle
 * Return true if @a P is inside of @a T
 *        false if @a P is not inside of @a T 
 * Complexity: O(1)
 */
template<typename TRIANGLE>
bool PointinTriangle(const TRIANGLE& T, const Point& P){
    auto A = T.node(0).position();
    auto B = T.node(1).position();
    auto C = T.node(2).position();
    Point v0 = C - A ;  // an edge of the triangle
    Point v1 = B - A ;  // an edge of the triangle
    Point v2 = P - A ;  // a vector from P to the triangle

    double dot00 = dot(v0, v0);
    double dot01 = dot(v0, v1) ;
    double dot02 = dot(v0, v2) ;
    double dot11 = dot(v1, v1) ;
    double dot12 = dot(v1, v2) ;

    double inverDeno = 1.0 / (dot00 * dot11 - dot01 * dot01) ;

    double u = (dot11 * dot02 - dot01 * dot12) * inverDeno ;
    if (u < 0 || u > 1) // if u out of range, return directly
    {
        return false ;
    }

    double v = (dot00 * dot12 - dot01 * dot02) * inverDeno ;
    if (v < 0 || v > 1) // if v out of range, return directly
    {
        return false ;
    }

    return (u + v <= 1) ;
}

/* Check if a point is on a line segment 
 * @param[in] q The point that we are checking
 * @param[in] p,r The two points representing the line segment
 * Return true if @a q is on line segment @a p to @a r.
 *        false otherwise
 * Complexity: O(1)
 */
bool onSegment(Point p, Point q, Point r)
{
    if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
        q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
       return true;
 
    return false;
}
 
/* To find orientation of ordered triplet (p, q, r).
 * @param[in] p, q, r Three points whose orientation we looking for here
 * Return 0 --> p, q and r are colinear
 *        1 --> Clockwise
 *        2 --> Counterclockwise
 * Complexity: O(1)
 */
int orientation(Point p, Point q, Point r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);
 
    if (val == 0) return 0;  // colinear
 
    return (val > 0)? 1: 2; // clock or counterclock wise
}

/* Check if two line segments interact
 * @param[in] p1, q1 Representing one line segment
 * @param[in] p2, q2 Representing another line segment
 * Return true if p1q1 intersects p2q2
 *        false otherwise
 * Complexity: O(1)
 */
bool doIntersect(Point p1, Point q1, Point p2, Point q2)
{
    // Find the four orientations needed for general and
    // special cases
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);
 
    // General case
    if (o1 != o2 && o3 != o4)
        return true;
 
    // Special Cases
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;
 
    // p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;
 
    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;
 
     // p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;
 
    return false; // Doesn't fall in any of the above cases
}






