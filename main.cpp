#include <igl/opengl/glfw/Viewer.h>
#include <igl/readSTL.h>
#include <igl/unproject_onto_mesh.h>

int main(int argc, char *argv[])
{
  // Inline mesh of a cube
  Eigen::MatrixXd V, N, C;
  Eigen::MatrixXi F;

  igl::readSTL("/home/todd/Documents/Workspace/maskProject/Data/JCHANG-S1-TA-M.stl", V, F, N);

  C = Eigen::MatrixXd::Constant(F.rows(), 3, 1);
  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;

  viewer.callback_mouse_down =
    [&V,&F, &C](igl::opengl::glfw::Viewer& viewer, int, int)->bool
  {
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
      viewer.core().proj, viewer.core().viewport, V, F, fid, bc))
    {
      // paint hit red
      C.row(fid) << 0, 1, 0;
      viewer.data().set_colors(C);
      Eigen::Vector3d p1, p2, p3, pc;
      Eigen::Vector3i triIndex = F.row(fid);
      p1 = V.row(triIndex.x());
      p2 = V.row(triIndex.y());
      p3 = V.row(triIndex.z());
      pc = p1 * bc.x() + p2 * bc.y() + p3 * bc.z();
      std::cout<<"Vertex position: " << pc << std::endl;
      return true;
    }
    return false;
  };
  
  viewer.data().set_mesh(V, F);
  viewer.data().set_colors(C);
  viewer.data().set_face_based(true);
  viewer.data().show_lines = true;
  viewer.launch();
}
