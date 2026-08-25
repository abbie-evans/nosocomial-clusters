import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

immuno_dynamics = pd.read_csv('Immuno.csv')
non_immuno_dynamics = pd.read_csv('Normal.csv')

# choose the values for 'frac' corresponding to integer 'Time' values from 0 to 100
immuno_dynamics = immuno_dynamics[immuno_dynamics['Time'].isin(range(101))]
immuno_dynamics = immuno_dynamics.set_index('Time')
immuno_dynamics = immuno_dynamics.reindex(range(101), method='ffill')
immuno_dynamics = immuno_dynamics.reset_index()
immuno_dynamics = immuno_dynamics[['Time', 'frac']]

non_immuno_dynamics = non_immuno_dynamics[non_immuno_dynamics['Time'].isin(range(101))]
non_immuno_dynamics = non_immuno_dynamics.set_index('Time')
non_immuno_dynamics = non_immuno_dynamics.reindex(range(101), method='ffill')
non_immuno_dynamics = non_immuno_dynamics.reset_index()
non_immuno_dynamics = non_immuno_dynamics[['Time', 'frac']]

plt.plot(immuno_dynamics['Time'], immuno_dynamics['frac'], label='Immunocompromised')
plt.plot(non_immuno_dynamics['Time'], non_immuno_dynamics['frac'], label='Immunocompetent')
plt.legend()
plt.show()


def gillespie_simulation():

    # ---------------------------------------------------------
    # Setup Parameters and Initial Conditions
    # ---------------------------------------------------------
    num_rooms = 8 # Number of rooms
    max_stage = 100 # Maximum disease stages (s = 1, 2, 3, ...)

    # Parameters from the equations
    beta_i = immuno_dynamics['frac'].values[:max_stage]*1/2
    beta_n = non_immuno_dynamics['frac'].values[:max_stage]*1/2
    epsilon = 0.5 # Relative transmission rate between patients and nurses (compared to patient-to-patient)
    lam = 1 # Infection stage progression rate
    eta = 0.1 # Relative transmission rate from nurse-to-nurse (compared to patient-to-patient) 
    xi = 0.5 # Relative susceptibility of immunocompromised individuals (compared to immunocompetent)

    # Initial population state
    # Patients: tracked independently per room (r)
    # Reduced number of immunocompromised patients
    # 4 patients total in each room
    S_i = {0: 1, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0}
    S_n = {0: 2, 1: 4, 2: 3, 3: 4, 4: 4, 5: 3, 6: 4, 7: 4}
    I_i = {r: {s: 0 for s in range(1, max_stage + 1)} for r in range(num_rooms)}
    I_i[0][1] = 1  # Room 0 starts with 1 patient at Stage 1
    I_n = {r: {s: 0 for s in range(1, max_stage + 1)} for r in range(num_rooms)}

    # Nurses: (move between rooms)
    N_nurse = 6 # Susceptible nurses
    M_nurse = {s: 0 for s in range(1, max_stage + 1)} # Nurses currently at each stage

    # Lists to store history for plotting
    history_time = [0.0]
    history_S_i0 = [S_i[0]]
    history_N_nurse = [N_nurse]
    history_M_nurse_total = [sum(M_nurse.values())]
    history_I_total = [sum(sum(I_i[r].values()) for r in range(num_rooms))]

    current_time = 0
    max_time = 100 # Time limit for simulation 

    # ---------------------------------------------------------
    # Gillespie Simulation Loop
    # ---------------------------------------------------------
    count = 0
    while current_time < max_time:
        events = []
        rates = []
        
        # --- Event 1: Patient Infection in Room r (S_i -> I_i[stage=1]) ---
        for r in range(num_rooms):
            if S_i[r] > 0:
                # Calculating the sum terms from your equation
                sum_patient_inf = sum(beta_i[s-1] * I_i[r][s] for s in range(1, max_stage + 1))
                sum_patient_inf_non_immuno = sum(beta_n[s-1] * I_n[r][s] for s in range(1, max_stage + 1))
                sum_nurse_m = sum(epsilon * beta_n[s-1] * M_nurse[s] for s in range(1, max_stage + 1))
                
                rate_inf = S_i[r] * (sum_patient_inf + sum_patient_inf_non_immuno + sum_nurse_m)
                if rate_inf > 0:
                    events.append(('PATIENT_INF', r))
                    rates.append(rate_inf)

        # --- Event 1b: Patient Infection in Room r (S_n -> I_n[stage=1]) ---
        for r in range(num_rooms):
            if S_n[r] > 0:
                # Calculating the sum terms from your equation
                sum_patient_inf = sum(beta_i[s-1] * I_i[r][s] for s in range(1, max_stage + 1))
                sum_patient_inf_non_immuno = sum(beta_n[s-1] * I_n[r][s] for s in range(1, max_stage + 1))
                sum_nurse_m = sum(epsilon * beta_n[s-1] * M_nurse[s] for s in range(1, max_stage + 1))
                
                rate_inf = S_n[r] * xi *(sum_patient_inf + sum_patient_inf_non_immuno + sum_nurse_m)
                if rate_inf > 0:
                    events.append(('PATIENT_INF_NON_IMMUNO', r))
                    rates.append(rate_inf)
                    
        # --- Event 2: Patient Progression (I_i[stage=s] -> I_i[stage=s+1]) ---
        for r in range(num_rooms):
            for s in range(1, max_stage):
                rate_prog = lam * I_i[r][s]
                if rate_prog > 0:
                    events.append(('PATIENT_PROG', r, s))
                    rates.append(rate_prog)

        # --- Event 2b: Patient Progression (I_n[stage=s] -> I_n[stage=s+1]) ---
        for r in range(num_rooms):
            for s in range(1, max_stage):
                rate_prog = lam * I_n[r][s]
                if rate_prog > 0:
                    events.append(('PATIENT_PROG_NON_IMMUNO', r, s))
                    rates.append(rate_prog)
                    
        # --- Event 3: Nurse Infection (N -> M[stage=1]) ---
        if N_nurse > 0:
            sum_room_inf = 0
            sum_room_inf_non_immuno = 0
            for r in range(num_rooms):
                sum_room_inf += sum(epsilon * beta_i[s-1] * I_i[r][s] for s in range(1, max_stage + 1))
                sum_room_inf_non_immuno += sum(epsilon * beta_n[s-1] * I_n[r][s] for s in range(1, max_stage + 1))
            sum_nurse_inf = sum(eta * beta_n[s-1] * M_nurse[s] for s in range(1, max_stage + 1))
            
            rate_nurse_inf = N_nurse * (sum_room_inf + sum_room_inf_non_immuno + sum_nurse_inf)
            if rate_nurse_inf > 0:
                events.append(('NURSE_INF', None))
                rates.append(rate_nurse_inf)

        # --- Event 4: Nurse Progression (M[stage=s] -> M[stage=s+1]) ---
        for s in range(1, max_stage):
            rate_nurse_prog = lam * M_nurse[s]
            if rate_nurse_prog > 0:
                events.append(('NURSE_PROG', s))
                rates.append(rate_nurse_prog)

        # Calculate system totals
        total_rate = sum(rates)
        if total_rate == 0:
            break # Outbreak ended or no possible transitions left

        # if everyone is infected, stop the simulation
        if all(S_i[r] == 0 and S_n[r] == 0 for r in range(num_rooms)) and N_nurse == 0:
            print("All patients and nurses have progressed to the final stage. Stopping simulation.", current_time)
            break

        # if individuals in stage with max beta have a random number less than the detection probability then stop the simulation
        peak_beta_i_idx = np.argmax(beta_i)
        peak_beta_n_idx = np.argmax(beta_n)
        detection_probability_i = 0.6 # Detection probability for immunocompromised
        detection_probability_n = 0.4 # Detection probability for immunocompetent
        if any(I_i[r][peak_beta_i_idx] > 0 for r in range(num_rooms)) and np.random.rand() < detection_probability_i:
            break
        if any(I_n[r][peak_beta_n_idx] > 0 for r in range(num_rooms)) and np.random.rand() < detection_probability_n:
            break

        # Calculate time step dt
        r1 = np.random.rand()
        dt = -np.log(r1) / total_rate
        current_time += dt
        
        # Pick exactly one event using the weighted probabilities
        probabilities = [r / total_rate for r in rates]
        chosen_idx = np.random.choice(len(events), p=probabilities)
        chosen_event = events[chosen_idx]
        
        # Execute the selected single event
        event_type = chosen_event[0]
        
        if event_type == 'PATIENT_INF':
            r = chosen_event[1]
            S_i[r] -= 1
            I_i[r][1] += 1
            count += 1
        elif event_type == 'PATIENT_INF_NON_IMMUNO':
            r = chosen_event[1]
            S_n[r] -= 1
            I_n[r][1] += 1
            count += 1
        elif event_type == 'PATIENT_PROG':
            r, s = chosen_event[1], chosen_event[2]
            I_i[r][s] -= 1
            I_i[r][s+1] += 1
        elif event_type == 'PATIENT_PROG_NON_IMMUNO':
            r, s = chosen_event[1], chosen_event[2]
            I_n[r][s] -= 1
            I_n[r][s+1] += 1
        elif event_type == 'NURSE_INF':
            N_nurse -= 1
            M_nurse[1] += 1
        elif event_type == 'NURSE_PROG':
            s = chosen_event[1]
            M_nurse[s] -= 1
            M_nurse[s+1] += 1

        # Record the step history (all rooms and overall nurses)
        history_time.append(current_time)
        history_S_i0.append(S_i[0])
        # history_I_total.append(sum(sum(stage.values()) for stage in I_i.values()) + sum(sum(stage.values()) for stage in I_n.values()))
        history_I_total.append(sum(sum(I_i[r].values()) for r in range(num_rooms)) + sum(sum(I_n[r].values()) for r in range(num_rooms)))
        # if latest value of history_I_total is less than the previous value, print a warning
        if len(history_I_total) > 1 and history_I_total[-1] < history_I_total[-2]:
            print("Warning: Total infected patients decreased from {} to {} at time {}".format(history_I_total[-2], history_I_total[-1], current_time))
        history_N_nurse.append(N_nurse)
        history_M_nurse_total.append(sum(M_nurse.values()))

    print("Simulation completed. Total events executed: {}".format(count))

    return history_time, history_S_i0, history_I_total, history_N_nurse, history_M_nurse_total


num_runs = 100
all_history_time = []
all_history_S_i0 = []
all_history_I_total = []
all_history_N_nurse = []
all_history_M_nurse_total = []

prob = []
for n in range(1, 11):
    counter = 0
    for run in tqdm(range(num_runs)):
        np.random.seed(run)  # Set seed for reproducibility
        history_time, history_S_i0, history_I_total, history_N_nurse, history_M_nurse_total = gillespie_simulation()
        all_history_time.append(history_time)
        all_history_S_i0.append(history_S_i0)
        all_history_I_total.append(history_I_total)
        all_history_N_nurse.append(history_N_nurse)
        all_history_M_nurse_total.append(history_M_nurse_total)
        # if all_history_I_total > n, add 1 to counter
        if all_history_I_total[-1][-1] + all_history_M_nurse_total[-1][-1] >= n:
            counter += 1
        plt.plot(history_time, history_I_total, label='Run {}'.format(run + 1))

    prob.append(counter/num_runs)

plt.legend()
plt.show()

plt.plot(range(1, 11), prob)
plt.xlabel(r'Size of cluster ($n$)')
plt.ylabel(r'Probability of >= $n$ cases')
plt.tight_layout()
plt.show()
