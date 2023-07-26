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
#define INITIAL_TEMPERATURE 20
#define DELTA_TEMPERATURE 120
// co ile kroków następuje zapis:
#define PRINT_MOD 5

#define F_NAME "energy.csv"

const real sigma2 = SIGMA * SIGMA;
const real sigma6 = sigma2 * sigma2 * sigma2;
const real sigma12 = sigma6 * sigma6;

void CHECK_CUDA_OK(const cudaError_t &status) {
  if (status != cudaSuccess) {
    cout << "CUDA fuckup: ";
    cout << cudaGetErrorString(status) << endl;
  }
}

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
void copy_to_gpu(Atom_t *cpu_atoms, Atom_t *gpu_atoms, int n_atoms) {
    CHECK_CUDA_OK(cudaMemcpy(gpu_atoms, cpu_atoms, n_atoms * sizeof(Atom_t), cudaMemcpyHostToDevice));
}
void copy_to_cpu(Atom_t *cpu_atoms, Atom_t *gpu_atoms, int n_atoms) {
    CHECK_CUDA_OK(cudaMemcpy(cpu_atoms, gpu_atoms, n_atoms * sizeof(Atom_t), cudaMemcpyDeviceToHost));
}

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


/**

GPU VERSION

*/

// Kernel v1:

__global__ void simulation_version_1(Atom_t *atoms, int n_atoms, real *potential, real RCUT, real baloon_radius, real BALOON_STIFNESS) {
    // Którym atomem jesteśmy?
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < n_atoms) {
        // zeruj wkład energii potencjalnej, zeruj siły działające na atom:
        atoms[index].fx = atoms[index].fy = atoms[index].fz = 0.0;
        potential[index] = 0.0;

        // przeiteruj się po pozostałych atomach:
        for (int j = 0; j < n_atoms; ++j) {
            if (index != j) {
                // Oblicz odległość między atomami:
                real r2 =
                    (atoms[index].x - atoms[j].x) * (atoms[index].x - atoms[j].x) +
                    (atoms[index].y - atoms[j].y) * (atoms[index].y - atoms[j].y) +
                    (atoms[index].z - atoms[j].z) * (atoms[index].z - atoms[j].z);

                if (r2 < RCUT * RCUT) {
                    real r4 = r2 * r2;
                    real r6 = r4 * r2;
                    real r12 = r6 * r6;

                    real delta_force_x = 12 * EPSILON *
                                        (sigma12 / r12 - sigma6 / r6) *
                                        (atoms[index].x - atoms[j].x) / r2;
                    real delta_force_y = 12 * EPSILON *
                                        (sigma12 / r12 - sigma6 / r6) *
                                        (atoms[index].y - atoms[j].y) / r2;
                    real delta_force_z = 12 * EPSILON *
                                        (sigma12 / r12 - sigma6 / r6) *
                                        (atoms[index].z - atoms[j].z) / r2;

                    // zaktualizuj energię potencjalną układu: ( dzielimy na 2,
                    // bo Vij = Vji);
                    potential[index] +=
                        EPSILON * (sigma12 / r12 - (2 * sigma6) / r6) / 2;
                    // aktualizuj siłę działającą na ity atom:
                    atoms[index].fx += delta_force_x;
                    atoms[index].fy += delta_force_y;
                    atoms[index].fz += delta_force_z;

                    // aktualizuj siłę działającą na jty atom:
                    // atoms[j].fx -= delta_force_x;
                    // atoms[j].fy -= delta_force_y;
                    // atoms[j].fz -= delta_force_z;
                }
            }
        }
        // Dodaj człon pochodzący od balonu:
        real r = sqrt(atoms[index].x * atoms[index].x + atoms[index].y * atoms[index].y +
                    atoms[index].z * atoms[index].z);
        if (r > baloon_radius) {
        // update energii:
        if (index %100 == 0) {
            printf("%d: before baloon: %lf\n", index, potential[index]);
        }
        potential[index] += BALOON_STIFNESS *
                                    (r - baloon_radius) *
                                    (r - baloon_radius) / 2;
        if (index%100 == 0) {
            printf("%d: after baloon: %lf", index, potential[index]);
        }
        // update sił:
        atoms[index].fx -=
            BALOON_STIFNESS * (r - baloon_radius) * atoms[index].x / r;
        atoms[index].fy -=
            BALOON_STIFNESS * (r - baloon_radius) * atoms[index].y / r;
        atoms[index].fz -=
            BALOON_STIFNESS * (r - baloon_radius) * atoms[index].z / r;
        }
    }
}

// Kernel v2:

__global__ void simulation_version_2(Atom_t *atoms, int n_atoms, real *fx_buffer, real *fy_buffer, real *fz_buffer,
real *potential_buffer, real RCUT) {
    // blockidx.y - ...
    // dimBlock = 32
    // t = threadIdx.x;
    // i = blockIdx.x * blockDim.x + t; nr gospodarza, host_index
    // j = blockIdx.y; nr przedziału [p, q-1]
    // p = j * N_THREADS;
    // q = min((j + 1) * N_THREADS, N_ATOMS);

    real potential_tmp = 0.0;
    real fx,fy,fz;
    fx=fy=fz=0.0;

    // współrzędne:
    // blockDim.x = 32, bloki jednowymiarowe.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int interval_start = blockIdx.y * blockDim.x;
    int interval_end = min((blockIdx.y + 1)*blockDim.x, n_atoms);

    // zadeklaruj pamięć współdzieloną:
    __shared__ double coordinates[3 * 32];

    // zapisz swoje współrzędne do pamięci:
    if (interval_start + threadIdx.x < interval_end) {
        coordinates[threadIdx.x] = atoms[interval_start + threadIdx.x].x;
        coordinates[blockDim.x + threadIdx.x] = atoms[interval_start + threadIdx.x].y;
        coordinates[2 * blockDim.x + threadIdx.x] = atoms[interval_start + threadIdx.x].z;
    }
    __syncthreads();

    // teraz w pamięci współdzielonej mamy współrzędne naszych gości.
    // policzmy oddziaływania gospodarza z gośćmi:
    
    for (int i = 0; i + interval_start < interval_end; ++i) {
        if (index != i + interval_start) {
            // Oblicz odległość między atomami:
            real r2 =
                (atoms[index].x - coordinates[i]) * (atoms[index].x - coordinates[i]) +
                (atoms[index].y - coordinates[blockDim.x + i]) * (atoms[index].y - coordinates[blockDim.x + i]) +
                (atoms[index].z - coordinates[2*blockDim.x + i]) * (atoms[index].z - coordinates[2*blockDim.x + i]);

            if (r2 < RCUT * RCUT) {
                real r4 = r2 * r2;
                real r6 = r4 * r2;
                real r12 = r6 * r6;

                real delta_force_x = 12 * EPSILON *
                                    (sigma12 / r12 - sigma6 / r6) *
                                    (atoms[index].x - coordinates[i]) / r2;
                real delta_force_y = 12 * EPSILON *
                                    (sigma12 / r12 - sigma6 / r6) *
                                    (atoms[index].y - coordinates[blockDim.x + i]) / r2;
                real delta_force_z = 12 * EPSILON *
                                    (sigma12 / r12 - sigma6 / r6) *
                                    (atoms[index].z - coordinates[2*blockDim.x + i]) / r2;

                // zaktualizuj energię potencjalną układu: ( dzielimy na 2,
                // bo Vij = Vji);
                potential_tmp +=
                    EPSILON * (sigma12 / r12 - (2 * sigma6) / r6) / 2;
                // aktualizuj siłę działającą na ity atom:
                fx += delta_force_x;
                fy += delta_force_y;
                fz += delta_force_z;
            }
        }
    }

    __syncthreads();

    if (index < n_atoms) {
        // e_lj_6_big[i * cnt + j] = tmp_e_lj_6;
        // e_lj_12_big[i * cnt + j] = tmp_e_lj_12;
        // fx_big[i * cnt + j] = tmp_fx;
        // fy_big[i * cnt + j] = tmp_fy;
        // fz_big[i * cnt + j] = tmp_fz;
        // cnt = liczba bloków.
        int buffer_index = index * gridDim.x + blockIdx.y;
        potential_buffer[buffer_index] = potential_tmp;
        fx_buffer[buffer_index] = fx;
        fy_buffer[buffer_index] = fy;
        fz_buffer[buffer_index] = fz;
    }
}

__global__ void sumator(int n_blocks, Atom_t *atoms, int n_atoms, real *potential, real *potential_buffer, real *fx_buffer, real *fy_buffer, real *fz_buffer) {
    real tmp_pot, tmp_fx, tmp_fy, tmp_fz;
    tmp_pot = tmp_fx = tmp_fy = tmp_fz = 0.0;

    int host = blockIdx.x + blockDim.x + threadIdx.x;

    if (host < n_atoms) {
        for (int i = 0; i < n_blocks; ++i) {
        int buffer_index = host * n_blocks + i;
        tmp_pot += potential_buffer[buffer_index];
        tmp_fx += fx_buffer[buffer_index];
        tmp_fy += fy_buffer[buffer_index];
        tmp_fz += fz_buffer[buffer_index];
        }

        atoms[host].fx = tmp_fx;
        atoms[host].fy = tmp_fy;
        atoms[host].fz = tmp_fz;
        potential[host] = tmp_pot;
    }
    
}


// Kernel v3:

__global__ void simulation_version_3(Atom_t *atoms, int n_atoms, real *potential_host_buffer, real *potential_guest_buffer, real *fx_host_buffer,
real *fx_guest_buffer, real *fy_host_buffer, real *fy_guest_buffer, real *fz_host_buffer, real *fz_guest_buffer) {
    int host = blockDim.x * blockIdx.x + threadIdx.x;
    int interval_start = blockIdx.y * blockDim.x;
    int interval_end = min((blockIdx.y + 1)*blockDim.x, n_atoms);
    // dolna przekątna nic nie robi:
    if (host >= interval_end) return;

    // zadeklaruj pamięć współdzieloną:
    __shared__ double coordinates[3 * 32];

    // zadeklaruj pamięć współdzieloną

    // zapisz swoje współrzędne do pamięci:
    if (interval_start + threadIdx.x < interval_end) {
        coordinates[threadIdx.x] = atoms[interval_start + threadIdx.x].x;
        coordinates[blockDim.x + threadIdx.x] = atoms[interval_start + threadIdx.x].y;
        coordinates[2 * blockDim.x + threadIdx.x] = atoms[interval_start + threadIdx.x].z;
    }
    __syncthreads();

    if (host < interval_end) {
        // oblicz oddziaływania z gośćmi:
        for (int i = interval_start; i < interval_end; ++i) {
            // atom (i + threadIdx.x) % blockDim.x jest gościem
            int guest = (i+threadIdx.x)%blockDim.x;
            if (host != guest + interval_start) {
                // Oblicz odległość między atomami:
                real r2 =
                    (atoms[host].x - coordinates[guest]) * (atoms[host].x - coordinates[guest]) +
                    (atoms[host].y - coordinates[blockDim.x + guest]) * (atoms[host].y - coordinates[blockDim.x + guest]) +
                    (atoms[host].z - coordinates[2*blockDim.x + guest]) * (atoms[host].z - coordinates[2*blockDim.x + guest]);
                if (r2 < RCUT*RCUT){
                    real r4 = r2 * r2;
                    real r6 = r4 * r2;
                    real r12 = r6 * r6;

                    real delta_force_x = 12 * EPSILON *
                                        (sigma12 / r12 - sigma6 / r6) *
                                        (atoms[index].x - coordinates[i]) / r2;
                    real delta_force_y = 12 * EPSILON *
                                        (sigma12 / r12 - sigma6 / r6) *
                                        (atoms[index].y - coordinates[blockDim.x + i]) / r2;
                    real delta_force_z = 12 * EPSILON *
                                        (sigma12 / r12 - sigma6 / r6) *
                                        (atoms[index].z - coordinates[2*blockDim.x + i]) / r2;

                    // zaktualizuj energię potencjalną układu:
                    potential_tmp +=
                        EPSILON * (sigma12 / r12 - (2 * sigma6) / r6);
                    // aktualizuj siłę działającą na ity atom:
                    fx += delta_force_x;
                    fy += delta_force_y;
                    fz += delta_force_z;

                    int guest_buffer_index = i * gridDim.x + ceil(n_atoms / gridDim.x);
                    fx_buffer[guest_buffer_index] -= delta_force_x;
                    fy_buffer[guest_buffer_index] -= delta_force_y;
                    fz_buffer[guest_buffer_index] -= delta_force_z;
                    potential_buffer[guest_buffer_index] = 0;
                }
            }

        }

        __syncthreads();

        if (index < n_atoms) {
            // e_lj_6_big[i * cnt + j] = tmp_e_lj_6;
            // e_lj_12_big[i * cnt + j] = tmp_e_lj_12;
            // fx_big[i * cnt + j] = tmp_fx;
            // fy_big[i * cnt + j] = tmp_fy;
            // fz_big[i * cnt + j] = tmp_fz;
            // cnt = liczba bloków.
            int buffer_index = index * gridDim.x + blockIdx.y;
            potential_buffer[buffer_index] = potential_tmp;
            fx_buffer[buffer_index] = fx;
            fy_buffer[buffer_index] = fy;
            fz_buffer[buffer_index] = fz;
        }
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

  // Kernel v1:
  cout << "Running kernel 1:\n";

  // malloc gpu arrays:
  Atom_t *gpu_atoms = NULL;
  CHECK_CUDA_OK(cudaMalloc((void**)&gpu_atoms, N_ATOMS * sizeof(Atom_t)));
  real *potential_gpu = NULL;
  CHECK_CUDA_OK(cudaMalloc((void**)&potential_gpu, N_ATOMS * sizeof(Atom_t)));
  real *potential_cpu = (real *)malloc(N_ATOMS * sizeof(real));

  for (int i = 0; i < N_ITERS; ++i) {
    update_velocity_by_half_step(atoms, N_ATOMS, SIMULATION_STEP);
    move_atoms(atoms, N_ATOMS, SIMULATION_STEP);
    if (i % ANIMATION_FRAMES == 0) {
      save_frame_to_animation_csv(atoms, N_ATOMS, animation_csv);
    }

    // CPU
    // potential_energy = update_atoms_forces_compute_potential_energy(
    //     atoms, N_ATOMS, BALOON_RADIUS, BALOON_STIFNESS, RCUT);
    
    // Version 1
    copy_to_gpu(atoms, gpu_atoms, N_ATOMS);
    dim3 threads(1024);
    dim3 blocks(N_ATOMS / 1024 + 1);
    simulation_version_1<<<blocks, threads>>>(gpu_atoms, N_ATOMS, potential_gpu, RCUT, BALOON_RADIUS, BALOON_STIFNESS);
    cudaDeviceSynchronize();
    copy_to_cpu(atoms, gpu_atoms, N_ATOMS);
    CHECK_CUDA_OK(cudaMemcpy(potential_cpu, potential_gpu, N_ATOMS * sizeof(real), cudaMemcpyDeviceToHost));

    potential_energy = 0.0;
    for (int k = 0; k < N_ATOMS; ++k) {
        potential_energy += potential_cpu[k];
    }

    if (i % PRINT_MOD == 0) {
      cout << compute_temperature(N_ATOMS, kinetic_energy) << ", kinetic: "<< kinetic_energy<< ", potential: " << potential_energy << "\n";
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

    // CPU
    // potential_energy = update_atoms_forces_compute_potential_energy(
    //     atoms, N_ATOMS, BALOON_RADIUS, BALOON_STIFNESS, RCUT);

    // Version 1
    copy_to_gpu(atoms, gpu_atoms, N_ATOMS);
    dim3 threads(1024);
    dim3 blocks(N_ATOMS / 1024 + 1);
    simulation_version_1<<<blocks, threads>>>(gpu_atoms, N_ATOMS, potential_gpu, RCUT, BALOON_RADIUS, BALOON_STIFNESS);
    cudaDeviceSynchronize();
    copy_to_cpu(atoms, gpu_atoms, N_ATOMS);
    CHECK_CUDA_OK(cudaMemcpy(potential_cpu, potential_gpu, N_ATOMS * sizeof(real), cudaMemcpyDeviceToHost));

    potential_energy = 0.0;
    for (int k = 0; k < N_ATOMS; ++k) {
        potential_energy += potential_cpu[k];
    }

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
