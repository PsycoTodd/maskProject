#include <igl/opengl/glfw/Viewer.h>
#include <igl/readSTL.h>
#include <igl/unproject_onto_mesh.h>
#include <iostream>
#include <igl/rotation_matrix_from_directions.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/bfs_orient.h>

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

Eigen::Quaterniond QuaternionRot(Eigen::Vector3d x1, Eigen::Vector3d y1, Eigen::Vector3d z1,
                                 Eigen::Vector3d x2, Eigen::Vector3d y2, Eigen::Vector3d z2) {

    using namespace Eigen;
    Matrix3d M = x1*x2.transpose() + y1*y2.transpose() + z1*z2.transpose();

    Matrix4d N;
    N << M(0,0)+M(1,1)+M(2,2)   ,M(1,2)-M(2,1)          , M(2,0)-M(0,2)         , M(0,1)-M(1,0),
         M(1,2)-M(2,1)          ,M(0,0)-M(1,1)-M(2,2)   , M(0,1)+M(1,0)         , M(2,0)+M(0,2),
         M(2,0)-M(0,2)          ,M(0,1)+M(1,0)          ,-M(0,0)+M(1,1)-M(2,2)  , M(1,2)+M(2,1),
         M(0,1)-M(1,0)          ,M(2,0)+M(0,2)          , M(1,2)+M(2,1)         ,-M(0,0)-M(1,1)+M(2,2);

    EigenSolver<Matrix4d> N_es(N);
    Vector4d::Index maxIndex;
    N_es.eigenvalues().real().maxCoeff(&maxIndex);

    Vector4d ev_max = N_es.eigenvectors().col(maxIndex).real();

    Quaterniond quat(ev_max(0), ev_max(1), ev_max(2), ev_max(3));
    quat.normalize();

    return quat;
}

Eigen::Matrix3d MatrixRot(Eigen::Vector3d x1, Eigen::Vector3d y1, Eigen::Vector3d z1,
                                 Eigen::Vector3d x2, Eigen::Vector3d y2, Eigen::Vector3d z2) {
                                 
  Eigen::Matrix3d ret;
  ret.col(0) = x2.transpose();
  ret.col(1) = y2.transpose();
  ret.col(2) = z2.transpose();                              
                                 
  return ret;                            
}

Eigen::Matrix3d getScaleFactor(const Eigen::MatrixXd& source, const float targetWidth, const float targetHeight)
{
  Eigen::Vector3d min = source.colwise().minCoeff();
  Eigen::Vector3d max = source.colwise().maxCoeff();
  float sourceWidth = max.x() - min.x();
  float sourceHeight = max.y() - min.y();
  float sourceDepth = max.z() - min.z();
  float sourceRatio = sourceHeight / sourceWidth;
  float targetRatio = targetHeight / targetWidth;
  float scaleFactor = sourceRatio < targetRatio ? targetWidth / sourceWidth : targetHeight / sourceHeight;
  Eigen::Matrix3d ret = Eigen::Matrix3d::Identity();
  ret *= scaleFactor;
  ret.row(2)[2] = 1.6 / sourceDepth;
  return ret;
}

Eigen::Matrix4d getCoordSysAlignment(const Eigen::MatrixXd& source, const float targetWidth, const float targetHeight,
                                     const Eigen::Vector3d& po0, const Eigen::Vector3d& px0, const Eigen::Vector3d& py0, 
                                     const Eigen::Vector3d& po1, const Eigen::Vector3d& px1, const Eigen::Vector3d& py1)
{
  Eigen::Vector3d min = source.colwise().minCoeff();
  Eigen::Vector3d max = source.colwise().maxCoeff();
  float sourceWidth = max.x() - min.x();
  float sourceHeight = max.y() - min.y();
  float sourceRatio = sourceHeight / sourceWidth;
  float targetRatio = targetHeight / targetWidth;
  float scaleFactor = sourceRatio < targetRatio ? targetWidth / sourceWidth : targetHeight / sourceHeight;

  Eigen::Affine3d T1, T2, ret = Eigen::Affine3d::Identity();
  T1 = T2 = ret;
  Eigen::Vector3d x1, y1, x2, y2;
  x1 = (px0 - po0).normalized();
  y1 = (py0 - po0).normalized();
  x2 = (px1 - po1).normalized();
  y2 = (py1 - po1).normalized();
  T1.linear() << x1, y1, x1.cross(y1);
  T2.linear() << x2, y2, x2.cross(y2);
  ret.linear() = T2.linear() * T1.linear().inverse();

  ret.pretranslate(po1);
  ret.scale(scaleFactor);
  return ret.matrix();
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
  ret.rotate(Eigen::AngleAxisd(-3.1415926 / 2.0, Eigen::Vector3d::UnitY()));
  //ret.rotate(Eigen::Quaterniond::FromTwoVectors(-Eigen::Vector3d::UnitZ(), normal));
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

  float fringeFactor = 0.5f;

  targetWidth = fmin(source.row(i1)[2] - source.row(i2)[2], source.row(i4)[2] - source.row(i3)[2]) * fringeFactor;
  targetHeight = fmin(source.row(i4)[1] - source.row(i1)[1], source.row(i3)[1] - source.row(i2)[1]) * fringeFactor;

  center = (source.row(i1) + source.row(i2) + source.row(i3) + source.row(i4)) / 4;

  normal = Eigen::Vector3d(source.row(i1) - source.row(i4)).cross(Eigen::Vector3d(source.row(i3) - source.row(i4)));
  normal.normalize();
}

void obtainRegionInfo(const Eigen::MatrixXd& source, const Eigen::Vector3i& index, 
                      float& targetWidth, float& targetHeight,
                      Eigen::Vector3d& center, Eigen::Vector3d& xpt, Eigen::Vector3d& zpt)
{
//    i3 --------- i2
//     \           /
//           i1 
  int i1, i2, i3;
  i1 = index[0];
  i2 = index[1];
  i3 = index[2];
  
  float fringeFactor = 0.6f;
  targetWidth = fabs(source.row(i3)[2] - source.row(i2)[2]) * fringeFactor;
  targetHeight = fabs(source.row(i3)[1] - source.row(i1)[2]) * fringeFactor;

  center = source.row(i1) * 1.3/3 + source.row(i2) * 0.85 / 3 + source.row(i3) * 0.85 / 3;

  Eigen::Vector3d res = Eigen::Vector3d(source.row(i2) - source.row(i3));
  xpt = res.normalized();
  Eigen::Vector3d normal = Eigen::Vector3d(source.row(i1) - source.row(i3)).cross(Eigen::Vector3d(source.row(i2) - source.row(i3)));
  zpt = normal.normalized();
}

Eigen::Vector3d getBboxCenter(const Eigen::MatrixXd& source)
{
  Eigen::Vector3d max = source.colwise().maxCoeff();
  Eigen::Vector3d min = source.colwise().minCoeff();
  return (max + min) * 0.5;
}

int main(int argc, char *argv[])
{
  // Inline mesh of a cube
  Eigen::MatrixXd V, N;
  Eigen::MatrixXd Vword, Nword;
  Eigen::MatrixXi F, Fword;

  igl::readSTL(argv[1], V, F, N);
  igl::readSTL(argv[2], Vword, Fword, Nword);


  Eigen::MatrixXd C(F.rows() + Fword.rows(), 3);
  C << Eigen::RowVector3d(0.2, 0.2, 0.2).replicate(F.rows(), 1),
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

  Eigen::Vector3d v1 = getBboxCenter(V);
  Eigen::Vector3d v2 = getBboxCenter(Vword);

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
  /*obtainRegionInfo(V, {497085, 497195, 497147, 497145}, regionW, regionH, center, normal);

  Eigen::Matrix4d aliMat = getAlignment(Vword, regionW, regionH, center, normal);*/

  Eigen::Vector3d po, px, pz;
  obtainRegionInfo(V, {465442, 465443, 465441}, regionW, regionH, po, px, pz);

  /*Eigen::Matrix4d aliMat = getCoordSysAlignment(Vword, regionW, regionH, 
                                                {0, 0, 0}, {1, 0, 0}, {0, 0, -1},
                                                po, px, py);

  Vword = Vword * aliMat.block<3, 3>(0, 0);
  Vword = Vword.rowwise() + aliMat.block<3, 1>(0, 3).transpose();*/

/*  po = {28.7528, 22.2583, -8.18973};
  px = {28.7533, 22.1463, -9.18003};
  py = {29.4612, 21.5863, -8.09132};
  regionW = 31.3926;
  regionH = 16.9876;*/

  Eigen::Quaterniond quad = QuaternionRot(Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ(),
                                          px, px.cross(pz), pz);

  Eigen::Matrix3d matx = igl::rotation_matrix_from_directions<double>(Eigen::Vector3d::UnitX(), px);
  Eigen::Matrix3d mat = igl::rotation_matrix_from_directions<double>(matx * Eigen::Vector3d::UnitZ(), pz);

  mat = MatrixRot(Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ(), px, pz.cross(px), pz);
 /* quad = Eigen::Quaterniond(Eigen::AngleAxisd(3.1415926 / 2.0, Eigen::Vector3d::UnitX())) * 
         Eigen::Quaterniond(Eigen::AngleAxisd(-3.1415926 / 2.0, Eigen::Vector3d::UnitY())) * quad;*/
  Vword = Vword * getScaleFactor(Vword, regionW, regionH);
  std::cout << "text property " << Vword.colwise().minCoeff() << " " << Vword.colwise().maxCoeff() << std::endl;
  Vword = Vword * mat.transpose();
  Eigen::Vector3d shift = (Eigen::RowVector3d::UnitZ() * mat.transpose() * (0.4)).transpose();
  Vword = Vword.rowwise() + (po + shift).transpose();



  

  Eigen::MatrixXd Vsum(V.rows() + Vword.rows(), V.cols());
  Vsum << V, Vword;
  Eigen::MatrixXi Fsum(F.rows() + Fword.rows(), F.cols());
  Fword = Fword.array() + V.rows(); 
  Fsum << F, Fword;


  viewer.data().set_mesh(Vsum, Fsum);
  viewer.data().set_colors(C);
  viewer.data().set_face_based(true);
  viewer.data().show_lines = true;
  //viewer.data().show_faces = false;

  viewer.data().add_edges(po.transpose(), (po + px * 10).transpose(), Eigen::RowVector3d(1, 0, 0));
  viewer.data().add_edges(po.transpose(), (po + pz * 10).transpose(), Eigen::RowVector3d(1, 0.2, 0));
  viewer.data().add_edges(po.transpose(), (po + (pz.cross(px)) * 10).transpose(), Eigen::RowVector3d(1, 0.5, 0));

  Eigen::Vector3d newX = Eigen::RowVector3d::UnitX() * mat;
  Eigen::Vector3d newZ = Eigen::RowVector3d::UnitZ() * mat;
  Eigen::Vector3d newY = newX.cross(newZ);
  viewer.data().add_edges(po.transpose(), (po + newX*30).transpose(), Eigen::RowVector3d(0, 0, 1));
  viewer.data().add_edges(po.transpose(), (po + newY * 30).transpose(), Eigen::RowVector3d(0, 0.2, 1));
  viewer.data().add_edges(po.transpose(), (po + newZ * 30).transpose(), Eigen::RowVector3d(0, 0.5, 1));
  viewer.launch();
}
