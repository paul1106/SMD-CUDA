/* -*-c++-*- SemiGlobalMatching - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* Describe	: main
*/

#include "stdafx.h"
#include "SemiGlobalMatching.h"
#include <chrono>
using namespace std::chrono;

// opencv library
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#pragma comment(lib,"opencv_world310d.lib")
#else
#pragma comment(lib,"opencv_world310.lib")
#endif

/**
 * \brief
 * \param argv 3
 * \param argc argc[1]:��Ӱ��·�� argc[2]: ��Ӱ��·�� argc[3]: ��С�Ӳ�[��ѡ��Ĭ��0] argc[4]: ����Ӳ�[��ѡ��Ĭ��64]
 * \param eg. ..\Data\cone\im2.png ..\Data\cone\im6.png 0 64
 * \param eg. ..\Data\Reindeer\view1.png ..\Data\Reindeer\view5.png 0 128
 * \return
 */
int main(int argv, char** argc)
{
    if (argv < 3) {
        std::cout << "�������٣�������ָ������Ӱ��·����" << std::endl;
        return -1;
    }

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    // ��ȡӰ��
    std::string path_left = argc[1];
    std::string path_right = argc[2];

    cv::Mat img_left_c = cv::imread(path_left, cv::IMREAD_COLOR);
    cv::Mat img_left = cv::imread(path_left, cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

    if (img_left.data == nullptr || img_right.data == nullptr) {
        std::cout << "��ȡӰ��ʧ�ܣ�" << std::endl;
        return -1;
    }
    if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
        std::cout << "����Ӱ��ߴ粻һ�£�" << std::endl;
        return -1;
    }


    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    const sint32 width = static_cast<uint32>(img_left.cols);
    const sint32 height = static_cast<uint32>(img_right.rows);

    // ����Ӱ��ĻҶ�����
    auto bytes_left = new uint8[width * height];
    auto bytes_right = new uint8[width * height];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            bytes_left[i * width + j] = img_left.at<uint8>(i, j);
            bytes_right[i * width + j] = img_right.at<uint8>(i, j);
        }
    }

    printf("Loading Views...Done!\n");

    // SGMƥ��������
    SemiGlobalMatching::SGMOption sgm_option;
    // �ۺ�·����
    sgm_option.num_paths = 8;
    // ��ѡ�ӲΧ
    sgm_option.min_disparity = argv < 4 ? 0 : atoi(argc[3]);
    sgm_option.max_disparity = argv < 5 ? 64 : atoi(argc[4]);
    // census��������
    sgm_option.census_size = SemiGlobalMatching::Census5x5;
    // һ���Լ��
    sgm_option.is_check_lr = true;
    sgm_option.lrcheck_thres = 1.0f;
    // Ψһ��Լ��
    sgm_option.is_check_unique = true;
    sgm_option.uniqueness_ratio = 0.99;
    // �޳�С��ͨ��
    sgm_option.is_remove_speckles = true;
    sgm_option.min_speckle_aera = 50;
    // �ͷ���P1��P2
    sgm_option.p1 = 10;
    sgm_option.p2_init = 150;
    // �Ӳ�ͼ���
    // �Ӳ�ͼ���Ľ�������ɿ��������̣���������䣬�����У�������
    sgm_option.is_fill_holes = false;

    printf("w = %d, h = %d, d = [%d,%d]\n\n", width, height, sgm_option.min_disparity, sgm_option.max_disparity);

    // ����SGMƥ����ʵ��
    SemiGlobalMatching sgm;

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    // ��ʼ��
	printf("SGM Initializing...\n");
    auto start = std::chrono::steady_clock::now();
    if (!sgm.Initialize(width, height, sgm_option)) {
        std::cout << "SGM��ʼ��ʧ�ܣ�" << std::endl;
        return -2;
    }
    auto end = std::chrono::steady_clock::now();
    auto tt = duration_cast<std::chrono::milliseconds>(end - start);
    printf("SGM Initializing Done! Timing : %lf s\n\n", tt.count() / 1000.0);

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    // ƥ��
	printf("SGM Matching...\n");
    start = std::chrono::steady_clock::now();
    // disparity���鱣�������ص��Ӳ���
    auto disparity = new float32[uint32(width * height)]();
    if (!sgm.Match(bytes_left, bytes_right, disparity)) {
        std::cout << "SGMƥ��ʧ�ܣ�" << std::endl;
        return -2;
    }
    end = std::chrono::steady_clock::now();
    tt = duration_cast<std::chrono::milliseconds>(end - start);
    printf("\nSGM Matching...Done! Timing :   %lf s\n", tt.count() / 1000.0);

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ��ʾ�Ӳ�ͼ
    // ע�⣬������Ʋ�����disp_mat�����ݣ�����������ʾ�ͱ������õġ��������Ҫ�������disparity����������ݣ��������ظ�����
    cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
    float min_disp = width, max_disp = -width;
    for (sint32 i = 0; i < height; i++) {
        for (sint32 j = 0; j < width; j++) {
            const float32 disp = disparity[i * width + j];
            if (disp != Invalid_Float) {
                min_disp = std::min(min_disp, disp);
                max_disp = std::max(max_disp, disp);
            }
        }
    }
    for (sint32 i = 0; i < height; i++) {
        for (sint32 j = 0; j < width; j++) {
            const float32 disp = disparity[i * width + j];
            if (disp == Invalid_Float) {
                disp_mat.data[i * width + j] = 0;
            }
            else {
                disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }

    // cv::imshow("�Ӳ�ͼ", disp_mat);
    cv::Mat disp_color;
    applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
    // cv::imshow("�Ӳ�ͼ-α��", disp_color);

    // ������
    std::string disp_map_path = argc[1]; disp_map_path += ".d.png";
    std::string disp_color_map_path = argc[1]; disp_color_map_path += ".c.png";
    cv::imwrite(disp_map_path, disp_mat);
    cv::imwrite(disp_color_map_path, disp_color);
    printf("Disparity maps saved to:\n  %s\n  %s\n", disp_map_path.c_str(), disp_color_map_path.c_str());

    // cv::waitKey(0);

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    // �ͷ��ڴ�
    delete[] disparity;
    disparity = nullptr;
    delete[] bytes_left;
    bytes_left = nullptr;
    delete[] bytes_right;
    bytes_right = nullptr;

    // system("pause");
    return 0;
}

