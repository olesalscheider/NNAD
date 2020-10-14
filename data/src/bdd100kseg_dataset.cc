/**************************************************************************
 * NNAD (Neural Networks for Automated Driving) training scripts          *
 * Copyright (C) 2019 FZI Research Center for Information Technology      *
 *                                                                        *
 * This program is free software: you can redistribute it and/or modify   *
 * it under the terms of the GNU General Public License as published by   *
 * the Free Software Foundation, either version 3 of the License, or      *
 * (at your option) any later version.                                    *
 *                                                                        *
 * This program is distributed in the hope that it will be useful,        *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 * GNU General Public License for more details.                           *
 *                                                                        *
 * You should have received a copy of the GNU General Public License      *
 * along with this program.  If not, see <https://www.gnu.org/licenses/>. *
 **************************************************************************/

#include <sstream>
#include <iomanip>

#include "utils.hh"

#include "bdd100kseg_dataset.hh"

Bdd100kSegDataset::Bdd100kSegDataset(bfs::path basePath, Mode mode)
{
    //m_fov = 50.0;
    m_fov = 56.25; // Because of aspect ratio fixup

    switch (mode) {
    case Mode::Train:
        m_groundTruthPath = basePath / bfs::path("seg") / bfs::path("labels") / bfs::path("train");
        m_leftImgPath = basePath / bfs::path("seg") / bfs::path("images") / bfs::path("train");
        m_groundTruthSubstring = std::string("_train_id.png");
        m_leftImgSubstring = std::string(".jpg");
        break;
    case Mode::Test:
        m_leftImgPath = basePath / bfs::path("seg") / bfs::path("images") / bfs::path("test");
        m_leftImgSubstring = std::string(".jpg");
        break;
    case Mode::Val:
        m_groundTruthPath = basePath / bfs::path("seg") / bfs::path("labels") / bfs::path("val");
        m_leftImgPath = basePath / bfs::path("seg") / bfs::path("images") / bfs::path("val");
        m_groundTruthSubstring = std::string("_train_id.png");
        m_leftImgSubstring = std::string(".jpg");
        break;
    default:
        CHECK(false, "Unknown mode!");
    }

    for (auto &entry : bfs::recursive_directory_iterator(m_leftImgPath)) {
        if (entry.path().extension() == ".jpg") {
            auto relativePath = bfs::relative(entry.path(), m_leftImgPath);
            std::string key = relativePath.string();
            key = key.substr(0, key.length() - m_leftImgSubstring.length());
            m_keys.push_back(key);
        }
    }
    std::sort(m_keys.begin(), m_keys.end());
}

static cv::Mat fixAspectRatio(cv::Mat src, int value)
{
    cv::Mat dst;
    int w = src.cols;
    int h = src.rows;
    int r = 2 * h - w;
    cv::copyMakeBorder(src, dst, 0, 0, 0, r, cv::BORDER_CONSTANT, value);
    return dst;
}

std::shared_ptr<DatasetEntry> Bdd100kSegDataset::get(std::size_t i)
{
    CHECK(i < m_keys.size(), "Index out of range");
    auto key = m_keys[i];
    auto result = std::make_shared<DatasetEntry>();
    auto leftImgPath = m_leftImgPath / bfs::path(key + m_leftImgSubstring);
    cv::Mat leftImg = cv::imread(leftImgPath.string());
    CHECK(leftImg.data, "Failed to read image " + leftImgPath.string());
    leftImg = fixAspectRatio(leftImg, 128);
    result->input.left = toFloatMat(leftImg);
    result->input.prevLeft = result->input.left;
    if (!m_groundTruthSubstring.empty()) {
        auto labelImgPath = m_groundTruthPath / bfs::path(key + m_groundTruthSubstring);
        cv::Mat labelImg = cv::imread(labelImgPath.string(), cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
        labelImg = fixAspectRatio(labelImg, 255);
        labelImg.convertTo(labelImg, CV_32S);
        CHECK(labelImg.data, "Failed to read image " + labelImgPath.string());
        result->gt.pixelwiseLabels = labelImg;
    }
    result->gt.bbDontCareAreas = cv::Mat(result->input.left.size(), CV_32SC1, cv::Scalar(m_boundingBoxDontCareLabel));
    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = true;
    result->metadata.horizontalFov = m_fov;
    result->metadata.key = key;
    return result;
}

