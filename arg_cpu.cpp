#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

using real = float;
using namespace std;

#define EPSILON 1.19
#define SIGMA 0.369
#define kB_CONSTANT 8.31e-3
#define ATOM_MASS 39.95
#define DEBUG 1
#define INITIAL_TEMPERATURE 70
#define DELTA_TEMPERATURE 100
// co ile kroków następuje zapis:
#define SAVE_MOD 2
#define PRINT_MOD 20

#define F_NAME "energy.csv"

const real sigma2 = SIGMA * SIGMA;
const real sigma6 = sigma2 * sigma2 * sigma2;
const real sigma12 = sigma6 * sigma6;

// void CHECK_CUDA_OK(const cudaError_t &status) {
//   if (status != cudaSuccess) {
//     cout << "CUDA fuckup: ";
//     cout << cudaGetErrorString(status) << endl;
//   }
// }

const real a_constant = SIGMA / sqrt(2);

const real velocity_max =
    sqrt(3 * kB_CONSTANT * INITIAL_TEMPERATURE / ATOM_MASS);

typedef struct {
  real x;
  real y;
  real z;
  real vx;
  real vy;
  real vz;
  real fx;
  real fy;
  real fz;
} Atom_t;

void save_frame_to_animation_csv(Atom_t *atoms, int n_atoms,
                                 FILE *animation_csv) {
  for (int i = 0; i < n_atoms; ++i) {
    fprintf(animation_csv, "(%lf,%lf,%lf);", atoms[i].x, atoms[i].y,
            atoms[i].z);
  }
  fprintf(animation_csv, "\n");
}

void dbg_print_atoms_positions(Atom_t *atoms, int n_atoms) {
  int n_boxes_3 = n_atoms / 4;
  int n_boxes = cbrt(n_boxes_3);

  for (int i = 0; i < n_atoms; ++i) {
    if (i % 4 == 0) {
      cout << "[";
    }
    cout << "(" << atoms[i].x << ", " << atoms[i].y << ", "
         << atoms[i].z << ")";
    if ((i + 1) % 4 == 0) {
      cout << "] ";
    }
    if ((4 * i) % (n_boxes) == 0) {
      cout << "\n";
    }
    if ((4 * i) % (n_boxes * n_boxes) == 0) {
      cout << "\n";
    }
  }
}

// TODO: implement for cuda version:
void copy_to_gpu(Atom_t *cpu_atoms, Atom_t *gpu_atoms, int n_atoms);
void copy_to_cpu(Atom_t *cpu_atoms, Atom_t *gpu_atoms, int n_atoms);

// Nadaj atomom współrzędne początkowe, wyzeruj środek masy.
void give_coordinates_and_translate_to_center(Atom_t *atoms,
                                              int BOX_SIZE) {
  int i = 0;
  for (int ix = 0; ix < BOX_SIZE; ++ix) {
    for (int iy = 0; iy < BOX_SIZE; ++iy) {
      for (int iz = 0; iz < BOX_SIZE; ++iz) {
        atoms[i].x = a_constant * (0.5 + ix * 2);
        atoms[i].y = a_constant * (0.5 + iy * 2);
        atoms[i++].z = a_constant * (0.5 + iz * 2);

        atoms[i].x = a_constant * (0.5 + ix * 2);
        atoms[i].y = a_constant * (1.5 + iy * 2);
        atoms[i++].z = a_constant * (1.5 + iz * 2);

        atoms[i].x = a_constant * (1.5 + ix * 2);
        atoms[i].y = a_constant * (0.5 + iy * 2);
        atoms[i++].z = a_constant * (1.5 + iz * 2);

        atoms[i].x = a_constant * (1.5 + ix * 2);
        atoms[i].y = a_constant * (1.5 + iy * 2);
        atoms[i++].z = a_constant * (0.5 + iz * 2);
      }
    }
  }

  // oblicz środek masy:
  real Mx = 0.0, My = 0.0, Mz = 0.0;
  for (int at = 0; at < i; at++) {
    Mx += atoms[at].x;
    My += atoms[at].y;
    Mz += atoms[at].z;
  }
  Mx = Mx / i;
  My = My / i;
  Mz = Mz / i;

  // wyzeruj środek masy.
  for (int at = 0; at < i; at++) {
    atoms[at].x -= Mx;
    atoms[at].y -= My;
    atoms[at].z -= Mz;
  }
}

// Nadaj atomom prędkości i oblicz całkowitą energię kinetyczną
// układu. Wyzeruj prędkość średnią układu.
real give_velocities_compute_initial_kinetic_energy(Atom_t *atoms,
                                                    int n_atoms) {
  real MVx = 0, MVy = 0, MVz = 0;
  real Scale = velocity_max * 2;
  real Ekin = 0;

  // nadaj atomom losowe prędkości, wyzeruj prędkość średnią:
  for (int i = 0; i < n_atoms; i++) {
    atoms[i].vx = Scale * (1.0 * rand() / RAND_MAX - 0.5);
    atoms[i].vy = Scale * (1.0 * rand() / RAND_MAX - 0.5);
    atoms[i].vz = Scale * (1.0 * rand() / RAND_MAX - 0.5);
    MVx += atoms[i].vx;
    MVy += atoms[i].vy;
    MVz += atoms[i].vz;
  }
  MVx = MVx / n_atoms;
  MVy = MVy / n_atoms;
  MVz = MVz / n_atoms;
  for (int i = 0; i < n_atoms; i++) {
    atoms[i].vx -= MVx;
    atoms[i].vy -= MVy;
    atoms[i].vz -= MVz;
  }

  // Oblicz całkowitą energię kinetyczną układu
  for (int i = 0; i < n_atoms; i++) {
    // Ekin += |v|^2
    Ekin += (atoms[i].vx * atoms[i].vx + atoms[i].vy * atoms[i].vy +
             atoms[i].vz * atoms[i].vz);
  }
  Ekin = Ekin * 0.5 * ATOM_MASS;
  return Ekin;
}

real update_atoms_forces_compute_potential_energy(
    Atom_t *atoms, int n_atoms, int baloon_radius,
    real BALOON_STIFNESS, real RCUT) {
  real total_potential_energy = 0.0;

  for (int i = 0; i < n_atoms; ++i) {
    // zaktualizuj siły działający na ity atom:
    // wyzeruj siły:
    atoms[i].fx = atoms[i].fy = atoms[i].fz = 0.0;
  }

  for (int i = 0; i < n_atoms; ++i) {
    for (int j = i + 1; j < n_atoms; ++j) {
      // Oblicz odległość między atomami:
      real r2 =
          (atoms[i].x - atoms[j].x) * (atoms[i].x - atoms[j].x) +
          (atoms[i].y - atoms[j].y) * (atoms[i].y - atoms[j].y) +
          (atoms[i].z - atoms[j].z) * (atoms[i].z - atoms[j].z);

      if (r2 < RCUT * RCUT) {
        real r4 = r2 * r2;
        real r6 = r4 * r2;
        real r12 = r6 * r6;

        real delta_force_x = 12 * EPSILON *
                             (sigma12 / r12 - sigma6 / r6) *
                             (atoms[i].x - atoms[j].x) / r2;
        real delta_force_y = 12 * EPSILON *
                             (sigma12 / r12 - sigma6 / r6) *
                             (atoms[i].y - atoms[j].y) / r2;
        real delta_force_z = 12 * EPSILON *
                             (sigma12 / r12 - sigma6 / r6) *
                             (atoms[i].z - atoms[j].z) / r2;

        // zaktualizuj energię potencjalną układu: ( dzielimy na 2,
        // bo Vij = Vji);
        total_potential_energy +=
            EPSILON * (sigma12 / r12 - (2 * sigma6) / r6);
        // aktualizuj siłę działającą na ity atom:
        atoms[i].fx += delta_force_x;
        atoms[i].fy += delta_force_y;
        atoms[i].fz += delta_force_z;
        // aktualizuj siłę działającą na jty atom:
        atoms[j].fx -= delta_force_x;
        atoms[j].fy -= delta_force_y;
        atoms[j].fz -= delta_force_z;
      }
    }

    // Dodaj człon pochodzący od balonu:
    real r = sqrt(atoms[i].x * atoms[i].x + atoms[i].y * atoms[i].y +
                  atoms[i].z * atoms[i].z);
    if (r > baloon_radius) {
      // update energii:
      total_potential_energy += BALOON_STIFNESS *
                                (r - baloon_radius) *
                                (r - baloon_radius) / 2;
      // update sił:
      atoms[i].fx -=
          BALOON_STIFNESS * (r - baloon_radius) * atoms[i].x / r;
      atoms[i].fy -=
          BALOON_STIFNESS * (r - baloon_radius) * atoms[i].y / r;
      atoms[i].fz -=
          BALOON_STIFNESS * (r - baloon_radius) * atoms[i].z / r;
    }
  }
  return total_potential_energy;
}

real compute_kinetic_energy(Atom_t *atoms, int n_atoms) {
  real kinetic_energy = 0.0;
  for (int i = 0; i < n_atoms; ++i) {
    real v2 = atoms[i].vx * atoms[i].vx + atoms[i].vy * atoms[i].vy +
              atoms[i].vz * atoms[i].vz;
    kinetic_energy += v2;
  }
  kinetic_energy = kinetic_energy * 0.5 * ATOM_MASS;
  return kinetic_energy;
}

void move_atoms(Atom_t *atoms, int n_atoms, real time_step) {
  for (int i = 0; i < n_atoms; ++i) {
    atoms[i].x += atoms[i].vx * time_step;
    atoms[i].y += atoms[i].vy * time_step;
    atoms[i].z += atoms[i].vz * time_step;
  }
}

void update_velocity_by_half_step(Atom_t *atoms, int n_atoms,
                                  real time_step) {
  for (int i = 0; i < n_atoms; ++i) {
    atoms[i].vx += atoms[i].fx * time_step / (2 * ATOM_MASS);
    atoms[i].vy += atoms[i].fy * time_step / (2 * ATOM_MASS);
    atoms[i].vz += atoms[i].fz * time_step / (2 * ATOM_MASS);
  }
}

real compute_temperature(int n_atoms, real kinetic_energy) {
  return 2 * kinetic_energy / (3 * n_atoms * kB_CONSTANT);
}

void update_velocity_by_half_step_with_heating(Atom_t *atoms,
                                               int n_atoms,
                                               real time_step,
                                               real kinetic_energy, int heating_iters) {
  real alpha =
      DELTA_TEMPERATURE /
      (heating_iters * compute_temperature(n_atoms, kinetic_energy));

  for (int i = 0; i < n_atoms; ++i) {
    atoms[i].vx += atoms[i].fx * time_step / (2 * ATOM_MASS);
    atoms[i].vx *= (1 + alpha);
    atoms[i].vy += atoms[i].fy * time_step / (2 * ATOM_MASS);
    atoms[i].vy *= (1 + alpha);
    atoms[i].vz += atoms[i].fz * time_step / (2 * ATOM_MASS);
    atoms[i].vz *= (1 + alpha);
  }
}

int main(int argc, char **argv) {
  FILE *CSV;
  FILE *animation_csv;
  CSV = fopen(F_NAME, "w");
  animation_csv = fopen("animation.csv", "w");
  if (argc != 10) {
    cout << "Usage: " << argv[0]
         << " BOX_SIZE SIMULATION_STEP RCUT BALOON_RADIUS "
            "BALOON_STIFNESS "
            "TERM HEAT COOL ANIMATION_FRAMES\n";
    exit(1);
  }
  const int BOX_SIZE = atoi(argv[1]);
  const real SIMULATION_STEP = atof(argv[2]);
  const real RCUT = atof(argv[3]);
  const real BALOON_RADIUS = atof(argv[4]);
  const real BALOON_STIFNESS = atof(argv[5]);
  const int N_ITERS = atoi(argv[6]);

  const int HEAT = atoi(argv[7]);
  const int COOL = atoi(argv[8]);

  const int ANIMATION_FRAMES = atoi(argv[9]);

  if (DEBUG) {
    printf("SIMULATION_STEP=%lf, N_ITERS=%i, HEAT=%i, COOL=%i\n",
           SIMULATION_STEP, N_ITERS, HEAT, COOL);
  }

  srand(time(NULL));

  const int N_ATOMS = 4 * BOX_SIZE * BOX_SIZE * BOX_SIZE;
  real kinetic_energy;
  real potential_energy;
  real total_energy;
  real temperature = INITIAL_TEMPERATURE;

  Atom_t *atoms = (Atom_t *)malloc(N_ATOMS * sizeof(Atom_t));

  give_coordinates_and_translate_to_center(atoms, BOX_SIZE);

  kinetic_energy =
      give_velocities_compute_initial_kinetic_energy(atoms, N_ATOMS);

  potential_energy = update_atoms_forces_compute_potential_energy(
      atoms, N_ATOMS, BALOON_RADIUS, BALOON_STIFNESS, RCUT);

  total_energy = kinetic_energy + potential_energy;

  cout << "total energy: " << total_energy << "\n";

  real energy_avg, energy_var, energy_stdev, energy_min, energy_max,
      energy_start, energy_end;
  real energy_sum = 0.0, energy2_sum = 0.0;

  kinetic_energy = compute_kinetic_energy(atoms, N_ATOMS);

  for (int i = 0; i < N_ITERS; ++i) {
    update_velocity_by_half_step(atoms, N_ATOMS, SIMULATION_STEP);
    move_atoms(atoms, N_ATOMS, SIMULATION_STEP);
    if (i % ANIMATION_FRAMES == 0) {
      save_frame_to_animation_csv(atoms, N_ATOMS, animation_csv);
    }
    potential_energy = update_atoms_forces_compute_potential_energy(
        atoms, N_ATOMS, BALOON_RADIUS, BALOON_STIFNESS, RCUT);

    if (i % PRINT_MOD == 0) {
      cout << compute_temperature(N_ATOMS, kinetic_energy) << "\n";
      fprintf(CSV, "%d, %lf\n", i, compute_temperature(N_ATOMS, kinetic_energy));
    }
    update_velocity_by_half_step(atoms, N_ATOMS, SIMULATION_STEP);
    kinetic_energy = compute_kinetic_energy(atoms, N_ATOMS);
    total_energy = kinetic_energy + potential_energy;
    if (i == 0) {
      energy_start = total_energy;
      energy_min = energy_max = total_energy;
    } else if (i == N_ITERS - 1) {
      energy_end = total_energy;
    }
    energy_sum += total_energy;
    energy2_sum += total_energy * total_energy;
    if (total_energy < energy_min) {
      energy_min = total_energy;
    }
    if (total_energy > energy_max) {
      energy_max = total_energy;
    }
  }

  energy_avg = energy_sum / N_ITERS;
  energy_var = energy2_sum / N_ITERS - energy_avg * energy_avg;
  energy_stdev = sqrt(energy_var);

  cout << "Energy: " << energy_avg << "+-" << energy_stdev << "\n";
  cout << "Energy min: " << energy_min << "\n";
  cout << "Energy max: " << energy_max << "\n";
  cout << "Energy start: " << energy_start << "\n";
  cout << "Energy end: " << energy_end << "\n";

  cout << "\n\nBegin Heating simulation\n\n";

  energy_sum = energy2_sum = 0.0;

  energy_min = energy_max = energy_start = energy_end;

  kinetic_energy = compute_kinetic_energy(atoms, N_ATOMS);

  // Heating simulation:
  for (int i = 0; i < HEAT; ++i) {
    update_velocity_by_half_step(
        atoms, N_ATOMS, SIMULATION_STEP);
    move_atoms(atoms, N_ATOMS, SIMULATION_STEP);
    if (i % ANIMATION_FRAMES == 0) {
      save_frame_to_animation_csv(atoms, N_ATOMS, animation_csv);
    }
    potential_energy = update_atoms_forces_compute_potential_energy(
        atoms, N_ATOMS, BALOON_RADIUS, BALOON_STIFNESS, RCUT);

    if (i % PRINT_MOD == 0) {
      cout << compute_temperature(N_ATOMS, kinetic_energy) << "\n";
      fprintf(CSV, "%d, %lf\n", N_ITERS + 1 + i, compute_temperature(N_ATOMS, kinetic_energy));
    }
    update_velocity_by_half_step_with_heating(
        atoms, N_ATOMS, SIMULATION_STEP, kinetic_energy, HEAT);
    kinetic_energy = compute_kinetic_energy(atoms, N_ATOMS);
    total_energy = kinetic_energy + potential_energy;
    if (i == 0) {
      energy_start = total_energy;
      energy_min = energy_max = total_energy;
    } else if (i == N_ITERS - 1) {
      energy_end = total_energy;
    }
    energy_sum += total_energy;
    energy2_sum += total_energy * total_energy;
    if (total_energy < energy_min) {
      energy_min = total_energy;
    }
    if (total_energy > energy_max) {
      energy_max = total_energy;
    }
  }

  free(atoms);
  fclose(CSV);
  fclose(animation_csv);

  energy_avg = energy_sum / N_ITERS;
  energy_var = energy2_sum / N_ITERS - energy_avg * energy_avg;
  energy_stdev = sqrt(energy_var);

  cout << "Energy: " << energy_avg << "+-" << energy_stdev << "\n";
  cout << "Energy min: " << energy_min << "\n";
  cout << "Energy max: " << energy_max << "\n";
  cout << "Energy start: " << energy_start << "\n";
  cout << "Energy end: " << energy_end << "\n";
}
