#pragma once

#include <iostream>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>


class Particle{
    public:
        Particle();

        float getWeight() const { return this->weight; }
        cv::Point getPosition() const { return this->position; }
        void setWeight(int weight) { this->weight = weight; }
        void setPosition(cv::Point &position) {this->position = position; }
    private:
        float weight;
        cv::Point position;
}


class Tracker{
    public:
        Tracker(const cv::Mat&, const cv::Rect&, const cv::Size&, int);
        void ParticlesInitialization();
        void TargetHistogram(int);
        float TargetModelLikelihood(cv::Mat&, int, int);
        void TargetMotionModel(int);
        void ParticlesResampling();

        std::vector<Particle> getParticles() const { return this->particles; }
        std::vector<cv::Point> getParticlesPosition() const{
            std::vector<cv::Point> particles_position;
            for(const auto& particle : this->particles)
                particles_position.push_back(particle.getPosition());

            return particles_position;
        }
        std::vector<float> getParticlesWeight() const{
            std::vector<cv::Point> particles_weight;
            for(const auto& particle : this->particles)
                particles_weight.push_back(particle.getWeight());

            return particles_weight;
        }
        void setParticles(std::vector<Particle> particles){
            this->particles = particles;
        }
        void setParticlesPosition(std::vector<cv::Point> &particles_position) {
            int i = 0;
            for(auto &particle : this->particles){
                particle.setPosition(partcicles_position.at(i));
                ++i;
            }
        }
        void setParticlesWeight(std::vector<cv::Point> &particles_weight) {
            int i = 0;
            for(auto &particle : this->particles){
                particle.setWeight(partcicles_weight.at(i));
                ++i;
            }
        }
    private:
        std::vector<Particle> particles;
        cv::Mat target;
        cv::Size img_bound;
        cv::Rect target_position;
        cv::Mat target_histogram;
}


Tracker::Tracker(const cv::Mat &target, const cv::Rect &target_position,
                 const cv::Size& img_bound, int nb_particles){
    this->target = target;
    this->target_position = target_position;
    this->img_bound = img_bound;
    for(int i = 0; i < nb_particles; ++i)
        particles.push_back(new Particle(0.0));
}


void Tracker::ParticlesInitialization(){
    float uniform_particle_weight = float(1/this->particles.size());
    std::random_devide rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> x(this->img_bound.size().x,
                                      this->img_bound.size().x
                                      + this->img_bound.size().width);
    std::uniform_int_distribution<> y(this->img_bound.size().y,
                                      this->img_bound.size().y
                                      + this->img_bound.size().height);
    for(auto &particle : particles){
        particle.setPosition(cv::Point(x(gen), y(gen)));
        particle.setWeight(uniform_particle_weight);
    }
}


void Tracker::TargetHistogram(int nb_histogram_bins){
    cv::Mat hist_b;
    cv::Mat hist_g;
    cv::Mat hist_r;
    cv::Mat tmp_mat;
    int channe_b[] = {0};
    int channe_g[] = {1};
    int channe_r[] = {2};
    int ranges = {0, 256};

    cv::calcHist(&this->target, 1, channels_b, cv::Mat(), hist_b, 2,
                 nb_histogram_bins, ranges);
    cv::calcHist(&this->target, 1, channels_g, cv::Mat(), hist_g, 2,
                 nb_histogram_bins, ranges);
    cv::calcHist(&this->target, 1, channels_r, cv::Mat(), hist_r, 2,
                 nb_histogram_bins, ranges);

    hist_b = hist_b.reshape(1, 1);
    hist_g = hist_g.reshape(1, 1);
    hist_r = hist_r.reshape(1, 1);

    cv::Scalar hist_b_sum = cv::sum(hist_b);
    cv::Scalar hist_g_sum = cv::sum(hist_g);
    cv::Scalar hist_r_sum = cv::sum(hist_r);
    cv::Scalar total = hist_b_sum + hist_g_sum + hist_r_sum;

    cv::hconcat(hist_b, hist_g, tmp_mat);
    cv::hconcat(tmp_mat, hist_r, this->target_histogram);

    this->target_histogram /= total;
}


float Tracker::TargetModelLikelihood(cv::Mat &candidate_histogram, int epsilon,
                                    int lambda){
    cv::Mat tmp_th = this->target_histogram + cv::Scalar(epsilon);
    cv::Mat tmp_ch = candidate_histogram + cv::Scalar(epsilon);
    cv::Mat div;
    cv::Mat log_div;

    cv::divide(tmp_th, tmp_ch, div);
    cv::log(div, log_div);

    float kullback_lieber_divergence = cv::sum(tmp_th.mul(log_div));
    float model_likelihood = std::exp(-lambda * kullback_lieber_divergence);

    return model_likelihood;
}


void Tracker::TargetMotionModel(int object_speed){
    std::random_devide rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> x(0, this->particles.size() - 1);
    std::uniform_int_distribution<> y(0, this->particles.size() - 1);
    std::vector<cv::Point> motion_estimation;

    for(int = 0; i < this->particles.size(); ++i)
        motion_estimation.push_back(
            cv::Scalar(object_speed) * cv::Point(x(gen), y(gen)));
    auto particles_position = this->getParticlesPosition()
    int i = 0;
    for(auto &particle_position : particles_position){
        particle_position += motion_estimation[i];
        ++i;
    }
    for(auto &particle_position : particles_position){
        particle_position.x = std::max(0, std::min(
            this->img_bound.size().x - this->target_position.size().x,
            motion_estimation[i].x));
        particle_position.y = std::max(0, std::min(
            this->img_bound.size().y - this->target_position.size().y,
            motion_estimation[i].y));
    }
    this->setParticlesPosition(particles_position);
}


void Tracker::ParticlesResampling(){
    auto particles_weight = this->getParticlesWeight();
    std::vector<float> weight_cumulated_sum;

    weight_cumulated_sum.push_back(particles_weight.at(0));
    for(int i = 1; i < particles_weight.size(): ++i)
        weight_cumulated_sum.push_back(
            particles_weight.at(i) + weight_cumulated.at(i - 1));

    std::random_devide rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> x(0, this->particles.size() - 1);
    std::vector<float> draws;
    std::vector<int> new_indexes(this->particles.size(), 0);

    for(int i = 0; i < this->particles.size(); ++i)
        draws.push_back(x(gen));

    for(int i = 0; i < draws.size(); ++i){
        for(int j = 0; j < weight_cumulated_sum.size(); ++j){
            if(weight_cumulated_sum.at(j) > draws.at(i)){
                new_indexes[i] = j;
                break;
            }
        }
    }

    std::vector<Particle> particles_resampled;
    for(const auto& new_index : new_indexes)
        particles_resampled.push_back(this->particles.at(new_index));

    this->setParticles(particles_resampled)
}

/*
class MeanShift{
    public:
        MeanShift();
        void TargetModelInitialisation(const cv::Mat&, const cv::Rect&, int, int);
        float EpanechnikovKernel(cv::Mat&);
        cv::Mat ProbabilityDensityFunction(const cv::Mat&, const cv::Rect&);
        cv::Mat ComputeWeight(const cv::Mat&, cv::Mat&, cv::Mat&, cv::Rect&);
        cv::Rect Track(const cv::Mat&, int);
    private:
        cv::Mat target_model;
        cv::Rect target_region;
        int maxIter;
        int nbHistogramBins;
};

MeanShift::MeanShift(){

}

void MeanShift::TargetModelInitialisation(const cv::Mat &frame, const cv::Rect &rect, int maxIter, int nbHistogramBins){
    this->target_region = rect;
    this->maxIter = maxIter;
    this->nbHistogramBins = nbHistogramBins;
    this->target_model = this->ProbabilityDensityFunction(frame, rect);
}


float MeanShift::EpanechnikovKernel(cv::Mat &kernel){
    float res;
    float sum = 0.0;
    int h = kernel.rows;
    int w = kernel.cols;

    for(int i = 0; i < h; ++i){
        for(int j = 0; j < w; ++j){
            float x = float(i - h/2);
            float y = float(j - w/2);
            float normalized_x = pow(x / (h/2), 2);
            float normalized_y = pow(y / (w/2), 2);
            float norm = normalized_x + normalized_y;
            if(norm < 1)
                res = 0.1*h*w*M_PI*(1.0 - norm);
            else
                res = 0.0;
            kernel.at<float>(i, j) = res;
            sum += res;
        }
    }
    return sum;
}


cv::Mat MeanShift::ProbabilityDensityFunction(const cv::Mat &frame, const cv::Rect &rect){
    int h = rect.height;
    int w = rect.width;
    int y = rect.y;
    int histogram_width = (256 / this->nbHistogramBins);
    cv::Mat pdf(3, histogram_width, CV_32F, cv::Scalar(1e-10));
    cv::Mat kernel = cv::Mat::zeros(cv::Size(h, w), CV_32F);
    float C = 1.0 / this->EpanechnikovKernel(kernel);

    for(int i = 0; i < h; ++i){
        int x = rect.x;
        for(int j = 0; j < w; ++j){
            float pu = C * kernel.at<float>(i, j);
            cv::Vec3f pixel = frame.at<cv::Vec3b>(y,x);
            auto r_bin = (pixel[0] / histogram_width);
            auto g_bin = (pixel[1] / histogram_width);
            auto b_bin = (pixel[2] / histogram_width);
            pdf.at<float>(0, r_bin) += pu;
            pdf.at<float>(1, g_bin) += pu;
            pdf.at<float>(2, b_bin) += pu;
            x++;
        }
        y++;
    }
    return pdf;
}


cv::Mat MeanShift::ComputeWeight(const cv::Mat &frame, cv::Mat &target_model, cv::Mat &target_candidate, cv::Rect &rect){
    int h = rect.height;
    int w = rect.width;
    int histogram_width = (256 / this->nbHistogramBins);
    cv::Mat weight(h, w, CV_32F, cv::Scalar(1.0));
    std::vector<cv::Mat> rgb_split;
    cv::split(frame, rgb_split);

    for(int channel = 0; channel < 3; ++channel){
        int y = rect.y;
        for(int i = 0; i < h; ++i){
            int x = rect.x;
            for(int j = 0; j < w; ++j){
                int histogram_bin_value = rgb_split[channel].at<unsigned char>(y, x) / histogram_width;
                float tm_val = target_model.at<float>(channel, histogram_bin_value);
                float tc_val = target_candidate.at<float>(channel, histogram_bin_value);
                weight.at<float>(i, j) *= float(sqrt(tm_val / tc_val));
                x++;
            }
        }
        y++;
    }
    return weight;
}


cv::Rect MeanShift::Track(const cv::Mat &frame, int epsilon){
    cv::Rect rect;
    for(int iteration = 0; iteration < this->maxIter; ++iteration){
        cv::Mat target_candidate = this->ProbabilityDensityFunction(frame, this->target_region);
        cv::Mat weight = this->ComputeWeight(frame, this->target_model, target_candidate, this->target_region);
        int x = weight.rows;
        int y = weight.cols;
        float sum_wij = 0.0;
        float sum_xwij = 0.0;
        float sum_ywij = 0.0;
        float center = float((x - 1) / 2.0);
        double coef = 0.0;

        rect.width = this->target_region.width;
        rect.height = this->target_region.height;
        rect.x = this->target_region.x;
        rect.y = this->target_region.y;

        for(int i = 0; i < x; ++i){
            for(int j = 0; j < y; ++j){
                float norm_i = float(i - center) / center;
                float norm_j = float(j - center) / center;
                if(pow(norm_i, 2) + pow(norm_j, 2) > 1)
                    coef = 0.0;
                else
                    coef = 1.0;
                sum_xwij += float(coef * norm_j * weight.at<float>(i, j));
                sum_ywij += float(coef * norm_i * weight.at<float>(i, j));
                sum_wij += float(coef * weight.at<float>(i, j));
            }
        }
        rect.x += int((sum_xwij / sum_wij) * center);
        rect.y += int((sum_ywij / sum_wij) * center);

        if(abs(rect.x - this->target_region.x) < epsilon &&
           abs(rect.y - this->target_region.y) < epsilon)
           break;
        else{
            this->target_region.x = rect.x;
            this->target_region.y = rect.y;
        }
    }
    return rect;
}*/