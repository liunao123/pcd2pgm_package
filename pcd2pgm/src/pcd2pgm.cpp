#include <ros/ros.h>

#include <nav_msgs/GetMap.h>
#include <nav_msgs/OccupancyGrid.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/filters/conditional_removal.h>         //条件滤波器头文件
#include <pcl/filters/passthrough.h>                 //直通滤波器头文件
#include <pcl/filters/radius_outlier_removal.h>      //半径滤波器头文件
#include <pcl/filters/statistical_outlier_removal.h> //统计滤波器头文件
#include <pcl/filters/voxel_grid.h>                  //体素滤波器头文件
#include <pcl/point_types.h>
#include <pcl/common/common.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/crop_box.h>

#include <Eigen/Core>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <png.h>

std::string png_file;
std::string file_directory;
std::string file_name;
std::string pcd_file;
double mean_z = 0;
std::string map_topic_name;
int auto_rotation = 0;

double x_min, x_max, y_min, y_max;

const std::string pcd_format = ".pcd";

nav_msgs::OccupancyGrid map_topic_msg;
//最小和最大高度
double thre_z_min = 0.3;
double thre_z_max = 2.0;
int flag_pass_through = 0;
double map_resolution = 0.05;
double thre_radius = 0.1;
//半径滤波的点数阈值
int thres_point_count = 10;

//直通滤波后数据指针
pcl::PointCloud<pcl::PointXYZ>::Ptr
    cloud_after_PassThrough(new pcl::PointCloud<pcl::PointXYZ>);
//半径滤波后数据指针
pcl::PointCloud<pcl::PointXYZ>::Ptr
    cloud_after_Radius(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr
    pcd_cloud(new pcl::PointCloud<pcl::PointXYZ>);

//直通滤波
void PassThroughFilter(const double &thre_low, const double &thre_high,
                       const bool &flag_in);
//半径滤波
void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pcd_cloud,
                         const double &radius, const int &thre_count);
//转换为栅格地图数据并发布
void SetMapTopicMsg(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                    nav_msgs::OccupancyGrid &msg);

void RotationPcdToHorizon(pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud);


void saveOccupancyGridMap2png(const nav_msgs::OccupancyGrid& occupancyGrid, const std::string& filename) ;

int main(int argc, char **argv) {
  ros::init(argc, argv, "pcl_filters");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  ros::Rate loop_rate(1.0);

  private_nh.param("file_name", file_name, std::string("/home/map.pcd"));

  pcd_file = file_name;
  std::cout << "pcd_file is : " << pcd_file << std::endl;

  private_nh.param("png_file", png_file, std::string("/home/map_test"));

  private_nh.param("thre_z_min", thre_z_min, 0.2);
  private_nh.param("thre_z_max", thre_z_max, 2.0);
  private_nh.param("flag_pass_through", flag_pass_through, 0);
  private_nh.param("thre_radius", thre_radius, 0.5);
  private_nh.param("map_resolution", map_resolution, 0.05);
  private_nh.param("thres_point_count", thres_point_count, 10);
  private_nh.param("map_topic_name", map_topic_name, std::string("map"));
  private_nh.param("auto_rotation", auto_rotation, 0);
  std::cout << "auto_rotation is : " << auto_rotation << std::endl;
  

  ros::Publisher map_topic_pub =
      nh.advertise<nav_msgs::OccupancyGrid>(map_topic_name, 10, true);

  // 下载pcd文件
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *pcd_cloud) == -1) {
    PCL_ERROR("Couldn't read file: %s \n", pcd_file.c_str());
    return (-1);
  }

  if ( bool( auto_rotation) )
  {
    RotationPcdToHorizon( pcd_cloud );
  }
  
  pcl::PointXYZ minPt, maxPt;
  pcl::getMinMax3D(*pcd_cloud, minPt, maxPt);
  x_min = minPt.x;
  x_max = maxPt.x;
  y_min = minPt.y;
  y_max = maxPt.y;

  std::cout << "minPt " << x_min << " " << y_min << " " << minPt.z << std::endl;
  std::cout << "maxPt " << x_max << " " << y_max << " " << maxPt.z << std::endl;

  if (minPt.x < 0 || minPt.y < 0 || minPt.z < 0)
  {
    std::cout << "Normal pcd to minPt ." << std::endl;
    for (int i = 0; i < pcd_cloud->points.size(); i++)
    {
      pcd_cloud->points[i].x -= minPt.x;
      pcd_cloud->points[i].y -= minPt.y;
      pcd_cloud->points[i].z -= minPt.z;
    }

    thre_z_min -= minPt.z;
    thre_z_max -= minPt.z;
    std::cout << " 136 normal thre_z_min and thre_z_max is: " << thre_z_min << " " << thre_z_max  << std::endl;

    auto normal_file = pcd_file;
    normal_file.insert(normal_file.size() - 4, "_normal");
    pcl::io::savePCDFileASCII(normal_file, *pcd_cloud);

    pcl::getMinMax3D(*pcd_cloud, minPt, maxPt);
    x_min = minPt.x;
    x_max = maxPt.x;
    y_min = minPt.y;
    y_max = maxPt.y;
    std::cout << "minPt and maxPt after Normal pcd to minPt ." << std::endl;
    std::cout << "minPt " << x_min << " " << y_min << " " << minPt.z << std::endl;
    std::cout << "maxPt " << x_max << " " << y_max << " " << maxPt.z << std::endl;
  }

  std::cout << "初始点云数据点数 " << pcd_cloud->points.size() << std::endl;
  //对数据进行直通滤波
  PassThroughFilter(thre_z_min, thre_z_max, bool(flag_pass_through));
  //对数据进行半径滤波
  RadiusOutlierFilter(cloud_after_PassThrough, thre_radius, thres_point_count);
  //转换为栅格地图数据并发布
  SetMapTopicMsg(cloud_after_Radius, map_topic_msg);

  ROS_WARN("start pub nav_msgs::OccupancyGrid topic . ");

  while (ros::ok()) {
    map_topic_pub.publish(map_topic_msg);

    loop_rate.sleep();

    // ros::spinOnce();
  }

  return 0;
}

//直通滤波器对点云进行过滤，获取设定高度范围内的数据
void PassThroughFilter(const double &thre_low, const double &thre_high,
                       const bool &flag_in) {
  // 创建滤波器对象
  pcl::PassThrough<pcl::PointXYZ> passthrough;
  //输入点云
  passthrough.setInputCloud(pcd_cloud);
  //设置对z轴进行操作
  passthrough.setFilterFieldName("z");
  //设置滤波范围
  passthrough.setFilterLimits(thre_low, thre_high);
  // true表示保留滤波范围外，false表示保留范围内
  passthrough.setFilterLimitsNegative(flag_in);
  //执行滤波并存储
  passthrough.filter(*cloud_after_PassThrough);
  // test 保存滤波后的点云到文件
  // pcl::io::savePCDFile<pcl::PointXYZ>(file_directory + "map_filter.pcd",
  //                                     *cloud_after_PassThrough);
  std::cout << "直通滤波后点云数据点数："
            << cloud_after_PassThrough->points.size() << std::endl;
  std::cout << "thre_low, thre_high："
            << thre_low << "    " << thre_high << std::endl;
}

//半径滤波
void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pcd_cloud0,
                         const double &radius, const int &thre_count) {
  //创建滤波器
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> radiusoutlier;
  //设置输入点云
  radiusoutlier.setInputCloud(pcd_cloud0);
  //设置半径,在该范围内找临近点
  radiusoutlier.setRadiusSearch(radius);
  //设置查询点的邻域点集数，小于该阈值的删除
  radiusoutlier.setMinNeighborsInRadius(thre_count);
  radiusoutlier.filter(*cloud_after_Radius);
  // test 保存滤波后的点云到文件
  // pcl::io::savePCDFile<pcl::PointXYZ>(file_directory + "map_radius_filter.pcd",
  //                                     *cloud_after_Radius);
  std::cout << "半径滤波后点云数据点数：" << cloud_after_Radius->points.size()
            << std::endl;
}

//转换为栅格地图数据并发布
void SetMapTopicMsg(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                    nav_msgs::OccupancyGrid &msg) {
  msg.header.seq = 0;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "base_link";

  msg.info.map_load_time = ros::Time::now();
  msg.info.resolution = map_resolution;

  // double x_min, x_max, y_min, y_max;
  double z_max_grey_rate = 0.05;
  double z_min_grey_rate = 0.95;
  //? ? ??
  double k_line =
      (z_max_grey_rate - z_min_grey_rate) / (thre_z_max - thre_z_min);
  double b_line =
      (thre_z_max * z_min_grey_rate - thre_z_min * z_max_grey_rate) /
      (thre_z_max - thre_z_min);

  if (cloud->points.empty()) {
    ROS_WARN("pcd is empty!\n");
    return;
  }

  // for (int i = 0; i < cloud->points.size() - 1; i++) {
  //   if (i == 0) {
  //     x_min = x_max = cloud->points[i].x;
  //     y_min = y_max = cloud->points[i].y;
  //   }

  //   double x = cloud->points[i].x;
  //   double y = cloud->points[i].y;

  //   if (x < x_min)
  //     x_min = x;
  //   if (x > x_max)
  //     x_max = x;

  //   if (y < y_min)
  //     y_min = y;
  //   if (y > y_max)
  //     y_max = y;
  // }
  // origin的确定
  msg.info.origin.position.x = x_min;
  msg.info.origin.position.y = y_min;
  msg.info.origin.position.z = 0.0;
  msg.info.origin.orientation.x = 0.0;
  msg.info.origin.orientation.y = 0.0;
  msg.info.origin.orientation.z = 0.0;
  msg.info.origin.orientation.w = 1.0;
  //设置栅格地图大小
  msg.info.width = int((x_max - x_min) / map_resolution);
  msg.info.height = int((y_max - y_min) / map_resolution);
  //实际地图中某点坐标为(x,y)，对应栅格地图中坐标为[x*map.info.width+y]
  msg.data.resize(msg.info.width * msg.info.height);

  // 设置成 0 就是 生成的图片 底色是 白色
  // 50 的话，在evo里面和背景色基本一致
  const unsigned char unkown_value = 50;
  msg.data.assign(msg.info.width * msg.info.height, unkown_value );

  ROS_INFO("data size = %d\n", msg.data.size());

  for (int iter = 0; iter < cloud->points.size(); iter++) {
    int i = int((cloud->points[iter].x - x_min) / map_resolution);
    if (i < 0 || i >= msg.info.width)
      continue;

    int j = int((cloud->points[iter].y - y_min) / map_resolution);
    if (j < 0 || j >= msg.info.height - 1)
      continue;
    // 栅格地图的占有概率[0,100]，这里设置为占据
    msg.data[i + j * msg.info.width] = 100;
    //    msg.data[i + j * msg.info.width] = int(255 * (cloud->points[iter].z *
    //    k_line + b_line)) % 255;
  }
  saveOccupancyGridMap2png(msg, png_file);
}


void RotationPcdToHorizon(pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud)
{
  // 创建一个模型参数对象，用于记录结果
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  // inliers表示误差能容忍的点 记录的是点云的序号
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  // 创建一个分割器
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional，这个设置可以选定结果平面展示的点是分割掉的点还是分割剩下的点。
  seg.setOptimizeCoefficients(true);
  // Mandatory-设置目标几何形状
  seg.setModelType(pcl::SACMODEL_PLANE);
  // 分割方法：随机采样法
  seg.setMethodType(pcl::SAC_RANSAC);
  // 设置误差容忍范围，也就是我说过的阈值
  seg.setDistanceThreshold(0.03);
  // 输入点云
  seg.setInputCloud( cloud );
  // 分割点云
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.size() == 0)
  {
    PCL_ERROR("Could not estimate a planar model for the given dataset. EXIT . ");
  }


  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  // Extract the inliers
  extract.setInputCloud (cloud);
  extract.setIndices (inliers);
  extract.setNegative (false);   //如果设为true,可以提取指定index之外的点云
  extract.filter (*cloud_p);
  pcl::io::savePCDFileASCII("/opt/csg/slam/navs/plan.pcd", *cloud_p);

  std::cout << "get planar model size: " << inliers->indices.size() << std::endl;

  std::cerr << "Model coefficients: " << coefficients->values[0] << " "
            << coefficients->values[1] << " "
            << coefficients->values[2] << " "
            << coefficients->values[3] << std::endl;

  // 参考: https://blog.csdn.net/weixin_38636815/article/details/109543753

  // 首先求解出旋转轴和旋转向量
  // a,b,c为求解出的拟合平面的法向量，是进行归一化处理之后的向量。
  Eigen::Vector3d plane_norm(coefficients->values[0], coefficients->values[1], coefficients->values[2]);

  // xz_norm是参考向量，也就是XOY坐标平面的法向量
  Eigen::Vector3d xz_norm(0.0, 0.0, 1.0);

  // 求解两个向量的点乘
  double v1v2 = plane_norm.dot(xz_norm);

  // 计算平面法向量和参考向量的模长，因为两个向量都是归一化之后的，所以这里的结果都是1.
  double v1_norm = plane_norm.norm();
  double v2_norm = xz_norm.norm();
  // 计算两个向量的夹角
  double theta = std::acos(v1v2 / (v1_norm * v2_norm));
  std::cout << "theta <rad> is  : " << theta << std::endl;
  std::cout << "theta <deg> is  : " << theta * 180.0 / 3.14159 << std::endl;

  // 根据向量的叉乘求解同时垂直于两个向量的法向量。
  Eigen::Vector3d axis_v1v2 = xz_norm.cross(plane_norm);

  // 对旋转向量进行归一化处理
  axis_v1v2 = axis_v1v2 / axis_v1v2.norm();

  // 计算旋转矩阵
  Eigen::AngleAxisd ro_vector(-theta, Eigen::Vector3d(axis_v1v2.x(), axis_v1v2.y(), axis_v1v2.z()));
  Eigen::Matrix3d ro_matrix = ro_vector.toRotationMatrix();
  // std::cout << "ro_matrix eigen " << ro_matrix << std::endl;

  pcl::PointCloud<pcl::PointXYZ> flat_cloud;
  flat_cloud = *cloud;
  flat_cloud.points.clear();

  printf("processing ");

  for (int i = 0; i < cloud->points.size(); i++)
  {
    Eigen::Vector3d newP(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
    Eigen::Vector3d new_point = ro_matrix * newP;
    pcl::PointXYZ pt;
    pt.x = new_point.x();
    pt.y = new_point.y();
    pt.z = new_point.z();
    flat_cloud.points.push_back(pt);
    int process = int(100 * double(i) / cloud->points.size());
    if (i % int(cloud->points.size() / 5) == 0)
      std::cout << process << "% ... " << std::endl;
  }
  std::cout << "100% ... " << std::endl;

  extract.setInputCloud ( flat_cloud.makeShared() );
  extract.setIndices (inliers);
  extract.setNegative (false);   //如果设为true,可以提取指定index之外的点云
  extract.filter (*cloud_p);
  auto plan_file = pcd_file;
  plan_file.insert(plan_file.size() - 4, "_plan");
  pcl::io::savePCDFileASCII(plan_file, *cloud_p);

  // 计算提取出的地面点的 平均高度，后面可以统一减去这个值
  double sum_z = 0;
  for (int i = 0; i < inliers->indices.size(); i++)
  {
    sum_z += flat_cloud.points[inliers->indices[i]].z;
  }
  mean_z = sum_z / inliers->indices.size();
  std::cout << "--------------------------------- " << std::endl;
  std::cout << "mean_z: " << mean_z << std::endl;
  std::cout << "--------------------------------- " << std::endl;
  // thre_z_max = mean_z + 5;
  // thre_z_min = mean_z - 5;

  for (int i = 0; i < cloud->points.size(); i++)
  {
    flat_cloud.points[i].z -= mean_z;
  }

  std::cout << "flat_cloud size: " << flat_cloud.points.size() << std::endl;

  pcd_file.insert(pcd_file.size() - 4, "_horizontal");
  pcl::io::savePCDFileASCII(pcd_file, flat_cloud);
  std::cout << "save result pcd :pcd_file_horizontal.pcd  >>  " << pcd_file << std::endl;

  cloud->points.clear();
  (*cloud) =  flat_cloud;
  std::cout << "--------------- rotation cloud size: " << cloud->points.size() << std::endl;

}

void saveOccupancyGridMap2png(const nav_msgs::OccupancyGrid& occupancyGrid, const std::string& filename) {
    int width = occupancyGrid.info.width;
    int height = occupancyGrid.info.height;

    std::string mapdatafile = filename + ".png";

    // 创建一个二维数组来保存占用地图数据
    std::vector<std::vector<uint8_t>> occupancyData(height, std::vector<uint8_t>(width, 255));

    // 将占用地图数据复制到二维数组中
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = j + (height - i - 1) * width;
            int8_t occupancyValue = occupancyGrid.data[index];
            // printf(" %d ", occupancyValue);
            if (occupancyValue == 0) {
                occupancyData[i][j] = 255; // 自由空间为白色
            } else if (occupancyValue == 100) {
                occupancyData[i][j] = 0; // 障碍物为黑色
            } 
            // else {
            //     occupancyData[i][j] = 200; // 未知区域为灰色 127
            // }
        }
    }

    // 打开PNG文件进行写入
    FILE* file = fopen(mapdatafile.c_str(), "wb");
    if (!file) {
        std::cout << "无法打开PNG文件" << std::endl;
        return;
    }

    // 初始化PNG结构
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cout << "无法创建PNG写入结构" << std::endl;
        fclose(file);
        return;
    }

    // 初始化PNG信息
    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cout << "无法创建PNG信息结构" << std::endl;
        png_destroy_write_struct(&png, nullptr);
        fclose(file);
        return;
    }

    // 设置PNG文件IO
    png_init_io(png, file);

    // 设置PNG图像信息
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    // 写入PNG头部
    png_write_info(png, info);

    // 写入PNG数据
    std::vector<png_bytep> rowPointers(height);
    for (int i = 0; i < height; ++i) {
        rowPointers[i] = reinterpret_cast<png_bytep>(occupancyData[i].data());
    }
    png_write_image(png, rowPointers.data());

    // 写入PNG结束
    png_write_end(png, nullptr);

    // 清理资源
    png_destroy_write_struct(&png, &info);
    fclose(file);

    std::cout << "占用地图已保存为PNG文件" << std::endl;

    // 写入对应的 yaml
    std::string mapmetadatafile = filename + ".yaml";
    printf("Writing map yaml to %s \n", mapmetadatafile.c_str());
    FILE* yaml = fopen(mapmetadatafile.c_str(), "w");

    geometry_msgs::Quaternion orientation = occupancyGrid.info.origin.orientation;
    Eigen::Quaterniond quaternion(orientation.w,orientation.x, orientation.y, orientation.z); // (w, x, y, z)
    // 将四元数转换为欧拉角表示
    Eigen::Vector3d euler = quaternion.toRotationMatrix().eulerAngles(2, 1, 0); // ZYX顺序

    // 获取欧拉角的yaw、pitch和roll值
    double yaw = euler[0];
    // double pitch = euler[1];
    // double roll = euler[2];

    fprintf(yaml, "image: %s\nresolution: %f\norigin: [%f, %f, %f]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n\n",
            mapdatafile.c_str(), occupancyGrid.info.resolution, occupancyGrid.info.origin.position.x, occupancyGrid.info.origin.position.y, yaw);

    fclose(yaml);
}