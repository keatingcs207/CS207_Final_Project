/**
 * @file Viewer_Extension.hpp
 *
 * Author: Zhenyang Pan & Yaxiong Cai
 */


#include "Mesh.hpp"


//***************//
//Listener struct//
//***************//

/** @class Pause_Listener
 * @brief Pause or unpause the visualizer
 * @right click: pause; right click again: unpause
 */
struct Pause_listener : public CS207::SDL_Listener {
  void handle(SDL_Event e) { 
    switch (e.type) {
      case SDL_MOUSEBUTTONDOWN: {
        if (e.button.button == SDL_BUTTON_RIGHT ) {
          if (dt_ == original_dt_)
            dt_ = 0;
          else
            dt_ = original_dt_;
        }
      } break;
    }
  }
  
  // constructor
  Pause_listener(double& dt) 
    : dt_(dt), original_dt_(dt) {}
  private:
   double& dt_;
   double original_dt_;
};




/** @class Speed_Listener
 * @brief Adjusts the simulation speed by keyboard.
 * @ Up: accelerate; Down: decelerate; Left: Recover the original speed
 */
struct Speed_listener : public CS207::SDL_Listener {
  private:
    double& dt_;
    double original_dt_;    
  
  public:  
    Speed_listener(double& dt, double  odt) 
      : dt_(dt), original_dt_(odt) {}

    virtual void handle(SDL_Event event) {
      if (event.type == SDL_KEYDOWN) { // keyboard key pressed
        switch (event.key.keysym.sym) {
          case SDLK_DOWN: // down arrow pressed
            dt_ -= 3e-4;
            break;
          case SDLK_UP: // up arrow pressed
            dt_ += 3e-4;
            break;
          case SDLK_LEFT: //recover dt
            dt_ = original_dt_;
            break;

          default: break; 
        }
      } 
  }
  
};  


/** @class XYZ_Listener
 * @brief Changes the position of the graph by keyboard
 * @ w, s: moving along x-axis; a, d: moving along y-axis
 */
template <typename M>
struct XYZ_listener : public CS207::SDL_Listener {

  virtual void handle(SDL_Event event) {

    if (event.type == SDL_KEYDOWN) { 
      switch (event.key.keysym.sym) {
        case SDLK_w: 
          for(auto b=(*mesh_).node_begin();b!=(*mesh_).node_end();++b){
            (*b).position().x += 0.5;
          }
          break;
        case SDLK_d:
          for(auto b=(*mesh_).node_begin();b!=(*mesh_).node_end();++b){
            (*b).position().y += 0.5;
          }
          break;
        case SDLK_s: 
          for(auto b=(*mesh_).node_begin();b!=(*mesh_).node_end();++b){
            (*b).position().x -= 0.5;
          }
          break;
        case SDLK_a:
          for(auto b=(*mesh_).node_begin();b!=(*mesh_).node_end();++b){
            (*b).position().y -= 0.5;
          }
          break;
        case SDLK_z: 
          for(auto b=(*mesh_).node_begin();b!=(*mesh_).node_end();++b){
            (*b).position().z += 0.5;
            std::cout<<(*b).position()<<std::endl;
          }
          break;
        case SDLK_x:
          for(auto b=(*mesh_).node_begin();b!=(*mesh_).node_end();++b){
            (*b).position().z -= 0.5;
            std::cout<<(*b).position()<<std::endl;
          }
          break;
          
        default: break; 
      }
    } 
  }
  
  XYZ_listener(M* mesh) 
    : mesh_(mesh){}
    
  private:
   M* mesh_;
};


/** @class Color_Listener
 * @brief Adjusts the color by keyboard.
 * @ Number key 1 2 3 4 5 6 7: pink, blue, red, cyan, yellow, green 
 */
struct Color_listener : public CS207::SDL_Listener {

  virtual void handle(SDL_Event event) {

    if (event.type == SDL_KEYDOWN) { 
      switch (event.key.keysym.sym) {
        case SDLK_1: 
          *a_ = 1;
          *b_ = 0;
          *c_ = 1;
          break;
        case SDLK_2: 
          *a_ = 0;
          *b_ = 0;
          *c_ = 1;
          break;
        case SDLK_3: 
          *a_ = 1;
          *b_ = 0;
          *c_ = 0;
          break;
        case SDLK_4: 
          *a_ = 0;
          *b_ = 1;
          *c_ = 1;
          break;
        case SDLK_5: 
          *a_ = 1;
          *b_ = 1;
          *c_ = 0;
          break;
        case SDLK_6: 
          *a_ = 0;
          *b_ = 1;
          *c_ = 0;
          break;
        case SDLK_7: 
          *a_ = 1;
          *b_ = 1;
          *c_ = 1;
          break;
          
        default: break; 
      }
    } 
  }
  
  Color_listener(int* a, int* b, int* c) 
    : a_(a), b_(b), c_(c){}
    
  private:
   int* a_;
   int* b_;
   int* c_;
};

// Color struct
struct color{
  int a_;
  int b_;
  int c_;
  color(int a, int b, int c): a_(a), b_(b), c_(c){}
  
  template <typename NODE>
  CS207::Color operator()(const NODE& n) {
    return CS207::Color(a_, b_, c_); 
  }
};


struct Force_listener : public CS207::SDL_Listener {
  private:
    int* a_;
    int* b_;    
    int* c_;
  
  public:  
    Force_listener(int* a, int* b, int* c): a_(a), b_(b), c_(c){}

    virtual void handle(SDL_Event event) {
      if (event.type == SDL_KEYDOWN) { // keyboard key pressed
        switch (event.key.keysym.sym) {
          case SDLK_f: 
            *a_ += 50;
            *b_ += 50;
            *c_ += 50;
            break;
          case SDLK_r: 
            *a_ = 0;
            *b_ = 0;
            *c_ = 0;
            break;

          default: break; 
        }
      } 
  }
  
};  

