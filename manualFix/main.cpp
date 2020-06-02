#include <igl/opengl/glfw/Viewer.h>
#include <igl/readSTL.h>
#include <igl/writeSTL.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <iostream>

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

Eigen::Matrix3d getScaleFactor(const Eigen::MatrixXd& source, const float targetWidth, const float targetHeight, const float targetDepth)
{
  Eigen::Vector3d min = source.colwise().minCoeff();
  Eigen::Vector3d max = source.colwise().maxCoeff();
  float sourceWidth = max.x() - min.x();
  float sourceHeight = max.y() - min.y();
  float sourceDepth = max.z() - min.z();
  float widthRatio = targetWidth / sourceWidth;
  float heightRatio = targetHeight / sourceHeight;
  float scaleFactor = heightRatio * sourceWidth > targetWidth ? widthRatio :heightRatio;
  Eigen::Matrix3d ret = Eigen::Matrix3d::Identity();
  ret *= scaleFactor;
  ret.row(2)[2] = targetDepth / sourceDepth;
  return ret;
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
  
  float fringeFactorWidth = 0.8;
  float fringeFactorHeight = 0.8;
  targetWidth = (source.row(i3) - source.row(i2)).norm() * fringeFactorWidth;
  targetHeight = (source.row(i3) - source.row(i1) + (source.row(i2) - source.row(i3)).normalized() * 
                 (source.row(i3) - source.row(i1)).dot((source.row(i3) - source.row(i2)).normalized())).norm() * fringeFactorHeight;

  center = source.row(i1) * 1.5/3 + source.row(i2) * 0.75 / 3 + source.row(i3) * 0.75 / 3;

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
  Eigen::Vector3i triIndex = {0, 0, 0}; // the side face triangle index
  Eigen::MatrixXd Vminus;
  Eigen::MatrixXi Fminus;
  int triI = 0; // used to keep track of the clicked point index.
  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;

  igl::readSTL(argv[1], V, F, N);
  igl::readSTL(argv[2], Vword, Fword, Nword);

  Eigen::MatrixXd Vsum(V.rows() + Vword.rows(), V.cols());
  Eigen::MatrixXi Fsum(F.rows() + Fword.rows(), F.cols());

  std::string outputFilePath = argv[3];


  Eigen::MatrixXd C(F.rows(), 3);
  C << Eigen::RowVector3d(0.2, 0.3, 0.8).replicate(F.rows(), 1);

  viewer.callback_mouse_down =
    [&V, &F, &C, &triI, &triIndex](igl::opengl::glfw::Viewer& viewer, int, int)->bool
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
      Eigen::Vector3i ids = F.row(fid);
      Eigen::Vector3d pv = getInterpolatedVector(V, ids, bc);
      int idx = getClosestIndex(ids, bc);
      triIndex[triI++%3] = idx;
      std::cout<<"**** Vertex position: " << pv.transpose() <<"  index: " << triIndex.transpose() << std::endl;
      return true;
    }
    return false;
  };

  viewer.callback_key_down = 
  [&V, &F, &C, &Vword, &Fword, &Vminus, &Fminus, &Vsum, &Fsum, &triIndex, &viewer, &outputFilePath]
  (igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)->bool
  {
    if (key == 'D')
    {
      float regionW, regionH;
      Eigen::Vector3d po, px, pz;
      Eigen::MatrixXd alignedVword;
      Eigen::MatrixXi alignedFword;
      obtainRegionInfo(V, triIndex, regionW, regionH, po, px, pz);

      Eigen::Matrix3d rotation;
      rotation.col(0) = px.transpose();
      rotation.col(1) = pz.cross(px).transpose();
      rotation.col(2) = pz.transpose();                              
      float wordDepth = 1.6;

      alignedVword = Vword * getScaleFactor(Vword, regionW, regionH, wordDepth) * rotation.transpose();
      Eigen::Vector3d shift = (Eigen::RowVector3d::UnitZ() * rotation.transpose() * wordDepth / 4).transpose(); // only leave 0.4 inside of the mask. 
      alignedVword = alignedVword.rowwise() + (po + shift).transpose();

      Vsum << V, alignedVword;
      alignedFword = Fword.array() + V.rows(); 
      Fsum << F, alignedFword;

      std::cout<<"############ Vsum row: " << Vsum.rows()<<std::endl;

      Eigen::MatrixXd C2(F.rows() + alignedFword.rows(), 3);
      C2 << Eigen::RowVector3d(0.2, 0.3, 0.8).replicate(F.rows(), 1),
            Eigen::RowVector3d(1.0, 0.7, 0.2).replicate(alignedFword.rows(), 1);
      C = C2; // so next time the color set is right.

/*
      igl::copyleft::cgal::mesh_boolean(V, F, alignedVword, alignedFword, igl::MESH_BOOLEAN_TYPE_UNION,Vminus,Fminus);
      Eigen::MatrixXd C2(Fminus.rows(), 3);
      C2 << Eigen::RowVector3d(0, 0.8, 0).replicate(Fminus.rows(), 1);
*/

      viewer.data().clear();
      viewer.data().set_mesh(Vsum, Fsum);
      viewer.data().set_colors(C2);
      return true;
    }

    else if (key == 'S')
    {
      igl::writeSTL(outputFilePath, Vsum, Fsum, false);
      std::cout << "Saved to " << outputFilePath <<std::endl;
      return true;
    }
    return true;
  };

  Eigen::Vector3d v1 = getBboxCenter(V);
  Eigen::Vector3d v2 = getBboxCenter(Vword);
  V = V.rowwise() - v1.transpose();
  Vword = Vword.rowwise() - v2.transpose();
 
  viewer.data().set_mesh(V, F);
  viewer.data().set_colors(C);
  viewer.data().set_face_based(true);
  viewer.data().show_lines = true;
  viewer.launch();
}
