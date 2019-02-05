import cv2
import random
import numpy as np


class Tracker():
    def __init__(self, image, target_position, nb_particles, object_speed):
        x, y, w, h = target_position
        self.target_pty = None
        self.target_ptx = None
        self.image = image
        self.target_image = image[y:y + h, x:x + w]
        self.target_histogram = None
        self.particles_weight = None
        self.particles_position = None
        self.particles_likelihood = None
        self.nb_particles = 200 if nb_particles is None else nb_particles
        self.object_speed = 70 if object_speed is None else object_speed

    #Compute the target BGR histogram
    def GetTargetHistogram(self):
        target_histogram_b = cv2.calcHist([self.target_image], [0], None, [255], [0,256]).flatten()
        target_histogram_g = cv2.calcHist([self.target_image], [1], None, [255], [0,256]).flatten()
        target_histogram_r = cv2.calcHist([self.target_image], [2], None, [255], [0,256]).flatten()
        target_histogram = np.concatenate((target_histogram_b, target_histogram_g, target_histogram_r))
        self.target_histogram = target_histogram / np.sum(target_histogram)

    #Initialize the particles with uniformely scatter position, uniform weight and 0 likelihood
    def ParticlesInitilization(self):
        y_random = np.random.uniform(0, self.image.shape[0], self.nb_particles)
        x_random = np.random.uniform(0, self.image.shape[1], self.nb_particles)
        self.particles_position = np.array([y_random, x_random]).T
        self.particles_weight = np.ones(self.nb_particles) / self.nb_particles
        self.particles_likelihood = np.zeros(self.nb_particles)

    #Resample the particles according to their weight
    def ParticlesResampling(self):
        particles_weight_cumulated_sum = np.cumsum(self.particles_weight)
        indexes = np.zeros(self.nb_particles, dtype=int)
        random_draws = np.random.rand(self.nb_particles)

        for i, random_draw in enumerate(random_draws):
            for j, weight in enumerate(particles_weight_cumulated_sum):
                if weight > random_draw:
                    indexes[i] = j
                    break

        self.particles_position = self.particles_position[indexes,:]
        self.particles_weight = self.particles_weight[indexes]

    #Apply the motion model and make sure that the new position are in the image boundaries
    def ParticlesMotionModel(self):
        self.particles_position += np.random.uniform(-self.object_speed, self.object_speed, self.particles_position.shape)
        self.particles_position[:, 0] = np.maximum(0, np.minimum(self.image.shape[0] - self.target_image.shape[0], self.particles_position[:, 0]))
        self.particles_position[:, 1] = np.maximum(0, np.minimum(self.image.shape[1] - self.target_image.shape[1], self.particles_position[:, 1]))

    #Compute the likelihood of a particle with the Kullback-Lieber divergence
    #computed on the local particle BGR histogram and on the candidate target BGR histogram
    def ParticleAppearanceModel(self, frame, particle_position):
        candidate_target_image = frame[int(particle_position[0]):int(particle_position[0]) + self.target_image.shape[0],
                                       int(particle_position[1]):int(particle_position[1]) + self.target_image.shape[1]]
        
        candidate_target_histogram_b = cv2.calcHist([candidate_target_image], [0], None, [255], [0,256]).flatten()
        candidate_target_histogram_g = cv2.calcHist([candidate_target_image], [1], None, [255], [0,256]).flatten()
        candidate_target_histogram_r = cv2.calcHist([candidate_target_image], [2], None, [255], [0,256]).flatten()
        candidate_target_histogram = np.concatenate((candidate_target_histogram_b, candidate_target_histogram_g, candidate_target_histogram_r))
        candidate_target_histogram /= np.sum(candidate_target_histogram)

        kullback_lieber_divergence = cv2.compareHist(self.target_histogram, candidate_target_histogram, cv2.HISTCMP_KL_DIV)
        particle_likelihood = np.exp(-kullback_lieber_divergence)

        return particle_likelihood

    #Compute the likelihood of every particles
    def ParticlesAppearanceModel(self, frame):
        for i in range(self.nb_particles):
            self.particles_likelihood[i] = self.ParticleAppearanceModel(frame, self.particles_position[i, :])

    #Update all particles weights by multiplying them by their respective likelihood
    #and normalize the newly computed particles weights
    def UpdateParticlesWeight(self):
        self.particles_weight *= self.particles_likelihood
        self.particles_weight /= np.sum(self.particles_weight)

    #Update target position by multiplying the old particle position
    #with the newly computed weight
    def UpdateTargetPosition(self):
        new_position = (self.particles_position.T @ self.particles_weight).astype(int)
        self.target_pty = new_position[0]
        self.target_ptx = new_position[1]

    def DrawTargetBox(self, frame):
        tl_box = (self.target_ptx, self.target_pty)
        br_box = (self.target_ptx + self.target_image.shape[1], self.target_pty + self.target_image.shape[0])
        cv2.rectangle(frame, tl_box, br_box, (0, 255, 0), 2)

    def DrawParticles(self, frame):
        for i in range(self.nb_particles):
            particle_x = self.particles_position[i, 1].astype(int) + self.target_image.shape[1] // 2
            particle_y = self.particles_position[i, 0].astype(int) + self.target_image.shape[0] // 2
            cv2.circle(frame, (particle_x, particle_y), 2, (0, 0, 255))