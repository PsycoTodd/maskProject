#include <igl/opengl/glfw/Viewer.h>
#include <igl/readSTL.h>
#include <igl/unproject_onto_mesh.h>
#include <iostream>

Eigen::Vector3d getInterpolatedVector(const Eigen::MatrixXd& V, const Eigen::Vector3i& ind, const Eigen::Vector3f& weight)
{
  Eigen::Vector3d p1, p2, p3, pc;
  p1 = V.row(ind.x());
  p2 = V.row(ind.y());
  p3 = V.row(ind.z());
  //std::cout<<p1.transpose().normalized() << "|" <<p2.transpose() << "|" << p3.transpose();
  pc = p1 * weight.x() + p2 * weight.y() + p3 * weight.z();
  return pc;
}

int getClosestIndex(const Eigen::Vector3i& ind, const Eigen::Vector3f& weight)
{
  float maxWeight = -1.f;
  int minIdx = -1;
  for(int i=0; i<weight.rows(); ++i) {
    if(weight[i] > maxWeight) {
      minIdx = ind[i];
      maxWeight = weight[i];
    }
  }
  return minIdx;
}

// We try to align the source mesh to a certain vector in the world.
Eigen::Matrix4d getAlignment(const Eigen::MatrixXd& source, const float targetWidth, const float targetHeight,
                             const Eigen::Vector3d& center, const Eigen::Vector3d& normal)
{
  Eigen::Vector3d min = source.colwise().minCoeff();
  Eigen::Vector3d max = source.colwise().maxCoeff();
  float sourceWidth = max.x() - min.x();
  float sourceHeight = max.y() - min.y();
  float sourceRatio = sourceHeight / sourceWidth;
  float targetRatio = targetHeight / targetWidth;
  float scaleFactor = sourceRatio < targetRatio ? targetWidth / sourceWidth : targetHeight / sourceHeight;

  Eigen::Affine3d ret = Eigen::Affine3d::Identity();
  // Translate
  ret.translate(center);
  // rotation
  ret.rotate(Eigen::Quaterniond::FromTwoVectors(-Eigen::Vector3d::UnitZ(), normal));
  ret.rotate(Eigen::AngleAxisd(3.1415926, Eigen::Vector3d::UnitY()));
  // scale first
  ret.scale(scaleFactor);
  return ret.matrix();
}

void obtainRegionInfo(const Eigen::MatrixXd& source, const Eigen::Vector4i& index, 
                      float& targetWidth, float& targetHeight,
                      Eigen::Vector3d& center, Eigen::Vector3d& normal)
{
//    i4 --------- i3
//    |            |
//    i1 --------- i2 

  int i1, i2, i3, i4; // from bottom left, go counter clock wise direction.
  i1 = index[0]; 
  i2 = index[1];
  i3 = index[2];
  i4 = index[3];

  float fringeFactor = 0.8f;

  targetWidth = fmin(source.row(i1)[0] - source.row(i2)[0], source.row(i4)[0] - source.row(i3)[0]) * fringeFactor;
  targetHeight = fmin(source.row(i4)[2] - source.row(i1)[2], source.row(i3)[2] - source.row(i2)[2]) * fringeFactor;

  center = (source.row(i1) + source.row(i2) + source.row(i3) + source.row(i4)) / 4;

  normal = Eigen::Vector3d(source.row(i1) - source.row(i4)).cross(Eigen::Vector3d(source.row(i3) - source.row(i4)));
  normal.normalize();
}

int main(int argc, char *argv[])
{
  // Inline mesh of a cube
  Eigen::MatrixXd V, N;
  Eigen::MatrixXd Vword, Nword;
  Eigen::MatrixXi F, Fword;

  igl::readSTL("/home/todd/Documents/Workspace/maskProject/Data/Batch1/Alexandra_Ke_D2_59_SFM-C-0.stl", V, F, N);
  igl::readSTL("/home/todd/Documents/Workspace/maskProject/Data/textModel/AlexandraK(D25).stl", Vword, Fword, Nword);

  Eigen::MatrixXd C(F.rows() + Fword.rows(), 3);
  C << Eigen::RowVector3d(0.2, 0.3, 0.8).replicate(F.rows(), 1),
       Eigen::RowVector3d(1.0, 0.7, 0.2).replicate(Fword.rows(), 1);
  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;

  viewer.callback_mouse_down =
    [&V, &N, &F, &C](igl::opengl::glfw::Viewer& viewer, int, int)->bool
  {
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
      viewer.core().proj, viewer.core().viewport, V, F, fid, bc))
    {
      // paint hit green
      C.row(fid) << 0, 1, 0;
      viewer.data().set_colors(C);
      Eigen::Vector3i triIndex = F.row(fid);
      Eigen::Vector3d pv = getInterpolatedVector(V, triIndex, bc);
      int idx = getClosestIndex(triIndex, bc);
      std::cout<<"Vertex position: " << pv <<"  "<< " closest index: " << idx << std::endl;
      return true;
    }
    return false;
  };

  Eigen::Vector3d v1 = V.colwise().mean();
  Eigen::Vector3d v2 = Vword.colwise().mean();

  V = V.rowwise() - v1.transpose();
  Vword = Vword.rowwise() - v2.transpose();

  // Obtain the mask region information.
  /*float regionWid = 29.9151f + 27.1758f;
  float regionHei = 43.4263f - 40.2759f;
  Eigen::Vector3d center = Eigen::Vector3d((29.9151 + 29.9307 -27.1758 - 27.1302) / 4,
                                           (18.4323 + 17.3042 + 18.4046 + 17.2646) / 4,
                                           (-40.2341 - 43.3516 - 40.2759 - 43.4263) / 4);
  Eigen::Vector3d normal = Eigen::Vector3d(29.9151 - 29.9307, 18.4323 - 17.3042, -40.2341 + 43.3516).
                           cross(Eigen::Vector3d(29.9151 + 27.1758, 18.4323 - 18.4046, -40.2341 + 40.2759));
  normal.normalize();*/

  float regionW, regionH;
  Eigen::Vector3d center, normal;
  obtainRegionInfo(V, {462066, 463776, 463777, 462068}, regionW, regionH, center, normal);

  Eigen::Matrix4d aliMat = getAlignment(V, regionW, regionH, center, normal);

  Vword = Vword * aliMat.block<3, 3>(0, 0);
  Vword = Vword.rowwise() + aliMat.block<3, 1>(0, 3).transpose();

  

  Eigen::MatrixXd Vsum(V.rows() + Vword.rows(), V.cols());
  Vsum << V, Vword;
  Eigen::MatrixXi Fsum(F.rows() + Fword.rows(), F.cols());
  Fword = Fword.array() + V.rows(); 
  Fsum << F, Fword;


  viewer.data().set_mesh(Vsum, Fsum);
  viewer.data().set_colors(C);
  viewer.data().set_face_based(true);
  viewer.data().show_lines = true;
  viewer.launch();
}
