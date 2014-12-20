/**
 * @file parachute_sim.cpp
 * Implementation of mass-spring system using Mesh
 *
 * @brief Simulates the movement of three objects suspended by different parachutes as they descend through the air.
 *  Models air resistance dynamically as the shape of the cloth-based parachutes is acted on by drag and changes shape. 
 * 
 */

#include <fstream>

#include "CS207/SDLViewer.hpp"
#include "CS207/Util.hpp"
#include "CS207/Color.hpp"
#include "Mesh.hpp"
#include "Point.hpp"
//Extensions
#include "jb_parallel.hpp"
#include "collision_parallel.hpp"


// Gravity in meters/sec^2
static constexpr double grav = 9.81;
static constexpr double K = 100.;

//Initial values for weights supported by parachutes
double w1 = 0.125;
double w2 = 0.25;
double w3 = 0.125;

/** Custom structure of data to store with Nodes */
struct NodeData {
  Point velocity;  //< Node velocity
  double mass;     //< Node mass
};

typedef Mesh<NodeData, double, int> MeshType;
typedef typename MeshType::node_type Node;
typedef typename MeshType::edge_type Edge;
typedef typename MeshType::triangle_type Triangle;


///////////////////////////////////////////////////
///						///
///	Viewer Extension & Interactivity	///
///						///
///////////////////////////////////////////////////


typedef typename CS207::SDLViewer Viewer;

//set simulation to be intially paused, wait for user command to start.
bool paused = true;

//Function that returns timestep dt or 0.0 depending on pause state
struct check_pause{
	double operator()(){
	if(paused){return 0.;}else{return dt_;}
}

check_pause(double dt):dt_(dt){}

double dt_;

};


//event handlers for keystrokes within the SDLViewer
struct test_key : public Viewer::EventListener {
  void operator()(const char* key) {
	switch(*key){

		//pause the simulation
		case 'p':
			if(paused == true){
					paused = false;
				}else{
					paused = true;
			}
			break;

		//increase or decrease the mass of each object
		case '1':
			w1 += 0.1;break;
		case '2':
			w2 += 0.1;break;
		case '3':
			w3 += 0.1;break;
		case '4':
			if(w1 - 0.1 > 0.)
			w1 -= 0.1;break;
		case '5':
			if(w2 - 0.1 > 0.)
			w2 -= 0.1;break;
		case '6':
			if(w3 - 0.1 > 0.)
			w3 -= 0.1;break;
		}	
	}
};



///////////////////////////////////////////////////
///						///
///	Viewer Extension & Interactivity	///
///						///
///////////////////////////////////////////////////



// Unary velocity functor for updating node velocities with a jb_parallel enabled symp euler step
template <typename F>
struct velocity_mod {
	F force;
	double dt;
	double t;
	void operator () (Node n) {
		n.value().value_.velocity += force(n, t) * (dt / n.value().value_.mass);
	}
	velocity_mod(F force1, double dt1, double t1) :
	force(force1), dt(dt1), t(t1) {};
};

// Unary position functor for updating node positions with a jb_parallel enabled symp euler step
template <typename F>
	struct position_mod {
	double dt;
	void operator () (Node n) {
	n.position() += n.value().value_.velocity * dt;
	}
	position_mod(double dt1) : dt(dt1) {};
};

/** Change a mesh's nodes according to a step of the symplectic Euler
 *    method with the given node force.
 * @param[in,out] m      Mesh
 * @param[in]     t      The current time (useful for time-dependent forces)
 * @param[in]     dt     The time step
 * @param[in]     force  Function object defining the force per node
 * @return the next time step (usually @a t + @a dt)
 *
 * @tparam G::node_value_type supports any struct with velocity and mass
 * @tparam F is a function object called as @a force(n, @a t),
 *           where n is a node of the mesh and @a t is the current time.
 *           @a force must return a Point representing the force vector on Node
 *           at time @a t.
 */
template <typename M, typename F, typename C>
double symp_euler_step(M& m, double t, double dt, F force, C constraints) {


//Create an instance of the position functor to be used with jb_parallel::for_each
position_mod<F> pm(dt);
jb_parallel::for_each(m.node_begin(),m.node_end(),pm);

  // Enfore constraints after position update but before force calculation.
  constraints(m, t);


// Contribute additional forces to represent weight hanging from the corners of each parachute:
//Apply forces from small weights hanging from corners of Chute 2:
m.node(0).value().value_.velocity += (Point(0,0, -grav * w1))* (dt / m.node(0).value().value_.mass);
m.node(12).value().value_.velocity += (Point(0,0, -grav * w1))* (dt / m.node(0).value().value_.mass);
m.node(4).value().value_.velocity += (Point(0,0, -grav * w1))* (dt / m.node(0).value().value_.mass);
m.node(8).value().value_.velocity += (Point(0,0, -grav * w1))* (dt / m.node(0).value().value_.mass);

//Apply forces from medium weights hanging from corners of Chute 2:
m.node(1217).value().value_.velocity += (Point(0,0, -grav * w2))* (dt / m.node(0).value().value_.mass);
m.node(1229).value().value_.velocity += (Point(0,0, -grav * w2))* (dt / m.node(0).value().value_.mass);
m.node(1221).value().value_.velocity += (Point(0,0, -grav * w2))* (dt / m.node(0).value().value_.mass);
m.node(1225).value().value_.velocity += (Point(0,0, -grav * w2))* (dt / m.node(0).value().value_.mass);

//Apply forces from small weights hanging from corners of Chute 3(Central Hole:
m.node(2434).value().value_.velocity += (Point(0,0, -grav * w3))* (dt / m.node(0).value().value_.mass);
m.node(2439).value().value_.velocity += (Point(0,0, -grav * w3))* (dt / m.node(0).value().value_.mass);
m.node(2444).value().value_.velocity += (Point(0,0, -grav * w3))* (dt / m.node(0).value().value_.mass);
m.node(2449).value().value_.velocity += (Point(0,0, -grav * w3))* (dt / m.node(0).value().value_.mass);


//Create an instance of the position functor to be used with jb_parallel::for_each
velocity_mod<F> vm(force, dt, t);
jb_parallel::for_each(m.node_begin(), m.node_end(), vm);


  return t + dt;
}

/** Set the direction flags on each triangle in order to 
 * keep track of the direction in which they are facing
 * @param[in] mesh, the current Mesh
 *
 * @pre: @a mesh is a valid mesh 
 * @post: for i in range(mesh), i.value() = -1 if i faces away from center and 1 otherwise
 *
 * Complexity: O(n)
 */
template <typename M>
void set_tri_directions(M& mesh) {
  // Set flags for surface normals
  // Approximate center of sphere
  Point sphere_center = Point(0,0,0);
  for (auto it = mesh.node_begin(); it != mesh.node_end(); ++it) {
    sphere_center += (*it).position();
  }
  sphere_center /= mesh.num_nodes();

  for(auto it = mesh.triangle_begin(); it != mesh.triangle_end(); ++it) {
    // Approximate center of triangle
    auto tri = (*it);
    Point tri_center = Point(0,0,0);
    for(unsigned i = 0; i < 3; ++i) {
      tri_center  += tri.node(i).position();
    }  
    tri_center /= 3.0;

   tri.value()=1;
  }
}


/** A boolean functor that compares two nodes
 *  based on their velocity values and returns true
 *  if the first node's velocity is smaller than the
 *  second one's */
struct VelComparator {
  template <typename Node>
  bool operator()(const Node& n1, const Node& n2) const {
    return (norm(n1.value().value_.velocity) < norm(n2.value().value_.velocity));
  }
};

/** A NodeColor functor that colors nodes based off
 *  of node velocities */
struct VelHeatMap {
  public:
    template <typename Node>
    // Returns a heat value for a node based off velocity(n)
    CS207::Color operator()(const Node& n) const {
      return CS207::Color::make_heat(norm(n.value().value_.velocity)/max_vel_);
    }
    /* Constructor */
    VelHeatMap(double max_vel) : max_vel_(max_vel) {}
  private:
    const double max_vel_;
};

///////////////////////////////////////////////////
///						///
///		FORCES				///
///						///
///////////////////////////////////////////////////

/** Air Pressure force functor for the mesh.
 * Return the force being applied to @a n at time @a t.
 */
struct DragForce {


  // Return the drag force acting on @a n at time @a t as a function of n.velocity.
  Point operator()(Node n, double t) {
    (void) t;
    Point n_i = Point(0,0,0);


   for(unsigned i = 0; i < n.value().neighbor_triangles_.size(); ++i) {
      auto tri = n.value().neighbor_triangle(i);
      n_i += (0.5*1.2*1.28) * //air resistance and drag coefficient constants
	 tri.area() * (-tri.surface_normal())/n.value().neighbor_triangles_.size();		 //scale the force for area orthogonal to motion and the number of triangles adjacent to the node

    }

//force modeled as linear drag, quadratic is unstable and blows up
	n_i *= n.value().value_.velocity;

    return n_i;
  }

};

/** Gravity force functor for the mesh.
 * Return the force being applied to @a n at time @a t.
 */
struct GravityForce {
  Point operator()(Node n, double t) {
    (void) t;
    return Point(0,0, -grav * n.value().value_.mass);
  }
};

/** Mass spring force functor for the mesh.
 * Return the force being applied to @a n at time @a t.
 */
struct MassSpringForce {
  Point operator()(Node n, double t) {
    (void) t;

    // initialize force
    Point force = Point(0,0,0);

    // calculate spring force
    for(auto it = n.edge_begin(); it != n.edge_end(); ++it) {
      MeshType::edge_type temp_edge = *it;
      MeshType::node_type temp_node = temp_edge.node2();
      Point diff = n.position() - temp_node.position();
      force += (-K) * (diff) / norm(diff) * (norm(diff) - temp_edge.value().value_);      
    }
    return force;
  }
};

/** Damping force functor for the mesh.
 * Return the force being applied to @a n at time @a t.
 */
struct DampingForce {
  double c_;

  // Constructor. Establishes constant for damping.
  DampingForce(const double& c) : c_(c) {};  

  // Return the damping force being applied to @a n at time @a t.
  Point operator()(Node n, double t) {
    (void) t;
    return -c_ * n.value().value_.velocity;
  }
};


/** CombinedForce construction as shown in class
 * Constructs a new force given two forces.
 * @param[in] force_1      First force
 * @param[in] force_2      Second force
 * 
 * @return a function object such output() = force_1_() + force_2_()
 *
 * @tparam force_1, force_2 are function objects called as @a force_#(n, @a t),
 *           where n is a node of the mesh and @a t is the current time.
 *           @a force must return a Point representing the force vector on Node
 *           at time @a t.
 */
template<typename F1, typename F2>
struct CombinedForce {
  Point operator()(Node n, double t) {
    return force_1_(n, t) + force_2_(n, t);
  }

  CombinedForce(F1 force_1, F2 force_2) 
    : force_1_(force_1), force_2_(force_2) {
  }

  private:
    F1 force_1_;
    F2 force_2_;
};

/** Combines two forces to create a new force
 *
 * @pre  Inputs can be called as @a fun(n, @a t),
 *        where n is a node and t is time.
 * Returns a CombinedForce
 */
template<typename F1, typename F2>
CombinedForce<F1, F2> make_combined_force(F1 force_1, F2 force_2) {
  return CombinedForce<F1, F2>(force_1, force_2);
}

/** Combines three forces to create a new force
 *
 * @pre  Inputs can be called as @a fun(n, @a t),
 *        where n is a node and t is time.
 * Returns a CombinedForce
 */
template<typename F1, typename F2, typename F3>
CombinedForce<F1, CombinedForce<F2, F3>> make_combined_force(F1 force_1, F2 force_2, F3 force_3) {
  CombinedForce<F2, F3> f2_f3_combo = CombinedForce<F2, F3>(force_2, force_3);
  return CombinedForce<F1, CombinedForce<F2, F3>>(force_1, f2_f3_combo);
}


///////////////////////////////////////////////////
///						///
///		CONSTRAINTS			///
///						///
///////////////////////////////////////////////////



/** Sets the x and y components of velocity equal to zero for the corner nodes acted on by the weight force.
*	Prevents canopy collapse at lower velocities.
 * */
template <typename M>
struct CornerConstraint {
  void operator()(M& m, double t) {
    (void) t;
std::vector<int> fixed_indexes;
fixed_indexes = {0,4,8,12,1217,1221,1225,1229,2434,2439,2444,2449};
    for(auto it = fixed_indexes.begin();it != fixed_indexes.end(); ++it) {
      m.node(*it).value().value_.velocity *= Point(0.,0.,1.);
      }
    
  }
};

//Boolean flags for whether a node from each chute has touched ground yet.
bool hit1 = false;
bool hit2 = false;
bool hit3 = false;
/** Constrains points to be above z = 0, and prints prints the time at which each parachute lands
 * O(N) Time
 */
template <typename M>
struct PlaneConstraint {

  void operator()(M& m, double t) {


//Check nodes in the first chute
    for(auto it = m.node_begin(); it != m.node_begin()+1217; ++it) {


      if (dot((*it).position(), Point(0,0,1)) < 0.) {
        (*it).position().z = 0;
		if(hit1==false){
			std::cout<<"Object 1 hits ground at "<<t<<" seconds!"<<"\n";
			hit1 = true;
		}
        (*it).value().value_.velocity.z = 0;

      }
    }


//Check Nodes in the second chute
    for(auto it = m.node_begin()+1217; it != m.node_begin()+2434; ++it) {



      if (dot((*it).position(), Point(0,0,1)) < 0.) {
        (*it).position().z = 0;
		if(hit2==false){
			std::cout<<"Object 2 hits ground at "<<t<<" seconds!"<<"\n";
			hit2 = true;
		}
        (*it).value().value_.velocity.z = 0;

      }
    }


//check nodes in the third chute
    for(auto it = m.node_begin()+2434; it != m.node_begin()+3585; ++it) {
      if (dot((*it).position(), Point(0,0,1)) < 0.) {
        (*it).position().z = 0;
		if(hit3==false){
			std::cout<<"Object 3 hits ground at "<<t<<" seconds!"<<"\n";
			hit3=true;
		}

        (*it).value().value_.velocity.z = 0;

      }
    }
  }
};



/** CombinedConstraint in the same vein as the CombinedForce
 * Constructs a new constraint given two constraints.
 * @param[in] constraint_1      First constraint
 * @param[in] constraint_2      Second constraint
 * 
 * @return a function object such output() = constraint_1_() + constraint_2_()
 *
 * @tparam constraint_1, constraint_2 are function objects called as @a pconstraint_#(g, @a t),
 *           where g is a mesh and @a t is the current time.
 *            no return value
 */
template<typename M, typename C1, typename C2>
struct CombinedConstraint {
  void operator()(M& m, double t) {
    constraint_1_(m, t);
    constraint_2_(m, t);
  }

  CombinedConstraint(C1 constraint_1, C2 constraint_2) 
    : constraint_1_(constraint_1), constraint_2_(constraint_2) {
  }

  C1 constraint_1_;
  C2 constraint_2_;
};

/** Combines two constraints to create a new constraint
 *
 * @pre  Inputs can be called as @a fun(g, @a t),
 *        where g is a node and t is time.
 * Returns a CombinedConstraint
 */
template<typename C1, typename C2, typename M = MeshType>
CombinedConstraint<M, C1, C2> make_combined_constraint(C1 constraint_1, C2 constraint_2) {
  return CombinedConstraint<M, C1, C2>(constraint_1, constraint_2);
}

/** Combines three constraints to create a new constraint
 *
 * @pre  Inputs can be called as @a fun(g, @a t),
 *        where g is a node and t is time.
 * Returns a CombinedConstraint
 */
template<typename C1, typename C2, typename C3, typename M = MeshType>
CombinedConstraint<M, C1, CombinedConstraint<M, C2, C3>> make_combined_constraint(C1 constraint_1, C2 constraint_2, C3 constraint_3) {
  CombinedConstraint<M, C2, C3> c2_c3_combo = CombinedConstraint<M, C2, C3>(constraint_2, constraint_3);
  return CombinedConstraint<M, C1, CombinedConstraint<M, C2, C3>>(constraint_1, c2_c3_combo);
}




int main(int argc, char** argv) {
  // Check arguments
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " NODES_FILE TETS_FILE\n";
    exit(1);
  }

  // Construct a mesh
  MeshType mesh;
  std::vector<typename MeshType::node_type> mesh_node;

  // Read all Points and add them to the Mesh
  std::ifstream nodes_file(argv[1]);
  Point p;
  while (CS207::getline_parsed(nodes_file, p)) {
    mesh_node.push_back(mesh.add_node(p));
  }

  // Read all mesh triangles and add them to the Mesh
  std::ifstream tris_file(argv[2]);
  std::array<int,3> t;
  while (CS207::getline_parsed(tris_file, t)) {
    mesh.add_triangle(mesh_node[t[0]], mesh_node[t[1]], mesh_node[t[2]]);
  }
  
  // Zero intial velocities.
  for(MeshType::node_iterator it = mesh.node_begin(); it != mesh.node_end(); ++it) {
    MeshType::node_type n = *it;
    n.value().value_.velocity = Point(0.,0.,0.);
    n.value().value_.mass = 0.25 / double(mesh.num_nodes());
  }

  // Set initial lengths to initial distances.
  for(MeshType::edge_iterator it = mesh.edge_begin(); it != mesh.edge_end(); ++it) {
    MeshType::edge_type e = *it;
      e.value().value_ = e.length();
  }

  // Print out the stats
  std::cout << mesh.num_nodes() << " " << mesh.num_edges() << std::endl;

  // Launch the SDLViewer
  CS207::SDLViewer viewer;
viewer.launch();
  auto node_map = viewer.empty_node_map(mesh);



  // ADD EVENT HANDLERS FOR SDL VIEWER

  test_key key=test_key();
  viewer.add_listener("KeyDown",&key);		



  viewer.add_nodes(mesh.node_begin(), mesh.node_end(), node_map);
  viewer.add_edges(mesh.edge_begin(), mesh.edge_end(), node_map);

  viewer.center_view();


  set_tri_directions(mesh);

  std::cout << "Starting\n";

  // Begin the mass-spring simulation
  double dt = 0.0001;
  double t_start = 0.0;
  double t_end   = 5.0;
  double max_vel = -UINT_MAX;
//create pause checker that stores initial dt
check_pause view_pause(dt);
  for (double t = t_start; t < t_end; t += dt) {
	dt = view_pause();//if paused == true, return dt = 0 to prevent sim from advancing
//create combined forces
	auto force = make_combined_force(
        make_combined_force(MassSpringForce(), 
                            GravityForce()),
        DragForce(),
        DampingForce(1/(mesh.num_nodes() * 20))   
    );
//create combined constraints
    auto constraints = make_combined_constraint(
        CornerConstraint<MeshType>(),
        PlaneConstraint<MeshType>()
        );
    symp_euler_step(mesh, t, dt, force, constraints);
    // Clear the viewer's nodes and edges.

    viewer.clear();
    node_map.clear();

    // Update the viewer with new node positions and color
    auto max_node = *std::max_element(mesh.node_begin(), 
                                                  mesh.node_end(), 
                                                  VelComparator());
    max_vel = std::max(max_vel, norm(max_node.value().value_.velocity));

    // Update viewer with nodes' new positions and new edges.
    viewer.add_nodes(mesh.node_begin(), mesh.node_end(), VelHeatMap(max_vel), node_map);
    viewer.add_edges(mesh.edge_begin(), mesh.edge_end(), node_map);

    viewer.set_label(t);


  }

  return 0;
}
