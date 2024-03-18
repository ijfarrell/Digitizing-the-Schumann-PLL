%SCHUMANN PLL DIGITAL AUDIO EFFECT ----------------------------------------
% This script is a digitization of the Schumann PLL Guitar Harmonizer
% developed by John Schumann in NYC in the early 2000s. The effect generates
% a square wave from an input audio signal (guitar etc.) and uses a
% Phase-locked Loop and decade counter IC to generate a higher harmonic
% square wave and a subsequent decade counter to generate a lower
% sub-harmonic. User may control volume levels of tones independently, as
% well as input gain, triggering, tracking, and output filtering values
% that determine the behavior of the effect.
%
% Developed by Isaiah Farrell as the dissertation project for the
% University of Edinburgh MSc in Acoustics and Music Technology.
%--------------------------------------------------------------------------

clc
clear
close all

% USER INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%INPUT PARAMETERS
INPUT_MODE = 1;                     % INPUT MODE SELECT: 1 = SOUND INPUT, 2 = SINE WAVE TEST TONE
ANALYSIS_MODE = 2;                  % ANALYSIS MODE SELECT: 1 = SPEED (ONLY OUTPUT SIGNAL FFT), 2 = IN-DEPTH (DECOMPOSED SIGNALS AND FILTER RESPONSES)
SOUNDOUT = 1;                       % PLAY SOUND 0 = NO SOUND, 1 = SOUND OUTPUT
SR = 192000;                        % SAMPLE RATE [192K "UP-SAMPLED" RECOMMENDED FOR EFFECT]

% MODE 1
FILENAME = "TESTSIG_LOWGUITAR_48k.wav";              % FILE FOR EFFECT TO BE APPLIED TO
OUTPUT_FILENAME = "RENDER_OUTPUT.wav";               % OUTPUT FILE NAME

% MODE 2
Tf = 3;                             % STOP TIME
f0 = 400;                           % TEST TONE FREQUENCY

% KNOBS AND SWITCHES
PREAMP      = 8;   %0-10            % INPUT PREAMPLIFICATION
TRIGGER     = 5.2;    %0-10         % TRIGGER THRESHOLD (-4V TO +4V)
LAGTIME     = 1;    %0-10           % PLL LAGTIME (R1 ON PLL)

RESPONSE    = .5;    %0-10          % LOOP FILTER CONTROL - DECREASES RESPONSE TIME
LOOPTRACK   = 3;    %0-10           % LOOP FILTER CONTROL - INCREASES TRACKING RATE

LOOPSPEED   = 1;    %1-3(INT)       % SPEED OF DECAY FASTEST-SLOWEST - INT VALUE 1,2,3
MULT_PHASE  = 0;     %0/1 (+/-)     % PHASE OF MULTIPLIED SQUARE WAVE 0=POS, 1=NEG
DIV_PHASE   = 0;    %0/1 (+/-)      % PHASE OF DIVIDED SQUARE WAVE 0=POS, 1=NEG

DIVIDER     = 5;    %0-10           % DIVIDED SIGNAL VOLUME
MULTIPLIER  = 3.5;  %0-10           % MULTIPLIED SIGNAL VOLUME
SQAREWAVE   = 4;    %0-10           % INPUT FREQ SIGNAL VOLUME
WAVESHAPE   = .3;   %0-10           % OUTPUT LOW-PASS FILTERING    
MASTER      = 10;   %0-10           % MASTER VOLUME

MULT_INTERVAL = 4;                  % MULTIPLIED FREQUENCY INTERVAL -
                                    % 1(UNISON), 2(OCTAVE), 3(OCTAVE+P5), 
                                    % 4(2OCT), 5(2OCT+M3), 6(2OCT+P5),
                                    % 7(2OCT+m7), 8(3OCT), 9(3OCT+M2)

DIV_INTERVAL = 5;                   % DIVIDED FREQUENCY INTERVAL -
                                    % 2(-OCTAVE), 3(-OCTAVE-P5), 4(-2OCT), 
                                    % 5(-2OCT-M3), 6(-2OCT-P5),7(-2OCT-m7), 
                                    % 8(-3OCT), 9(-3OCT-M2)


%END OF USER INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST INPUT SETTINGS ----------------------------------------------------
k = 1/SR;                                                                   % TIMESTEP
if INPUT_MODE == 1                                                          % INPUT SOUND MODE
    [test_sound,Testsound_Fs] = audioread(FILENAME);                        % READ INPUT FILE
    [Numer, Denom] = rat(SR/Testsound_Fs);
    test_sound = resample(test_sound, Numer, Denom);
    %test_sound = resample(test_sound,round(SR/Testsound_Fs),1);            % RESAMPLE TEST SOUND TO SAMPLE RATE
    Nf = length(test_sound);                                                % CALCULATE NUMBER OF SAMPLES
    t = (0:k:(Nf/SR)-k)';                                                   % CALCULATE TIME VECTOR
    x_in_test = test_sound(:,1);                                            % TEST SOUND (APPLY D/A INPUT VOLTAGE CONVERSION APPROX 500mV PEAK GUITAR SIGNAL VOLTAGE)
else
    Nf = floor(Tf*SR);                                                      % CALCULATE NUMBER OF SAMPLES
    t = (0:k:(Nf/SR)-k)';                                                   % CALCULATE TIME VECTOR
    x_in_test = sin(2*pi*f0.*t);                                            % GENERATE INPUT TEST SINE SIGNAL
end

x_in = [0;x_in_test];                                                       % BUFFER TEST AUDIO FOR LOOP

tvec = (0:Nf-1)'*k ;                                                        % TIME VECTOR FOR PLOTS
fvec = (0:Nf-1)'*SR/Nf;                                                     % FREQUENCY VECTOR FOR PLOTS
% ------------------------------------------------------------------------

% INPUT CONDITIONING SETTINGS --------------------------------------------
tic
%OPEN LOOP SPECS
BIAS = (TRIGGER-5)*(.8); %+/- 4V                                            % CALCULATE TRIGGER VOLTAGE                          
AOL = 125; %dB                                                              % DEFINE OPEN LOOP OP AMP GAIN
SWEEP_HI_ol = 11.85;                                                        % DEFINE UPPER SUPPLY VOLTAGE
SWEEP_LO_ol = -11.85;                                                       % DEFINE LOWER SUPPLY VOLTAGE

% SCHMITT TRIGGER SPECS
V_HI = 6.5;                                                                 % DEFINE HIGH TRIGGER VOLTAGE
V_LO = 4.5;                                                                 % DEFINE LOW TRIGGER VOLTAGE
SWEEP_HI_st = 1;                                                            % DEFINE HIGH VOLTAGE (ARBITRARY - USE 1 BIT AS SQUARE WAVE PLACEHOLDER)
SWEEP_LO_st = 0;                                                            % DEFINE LOW VOLTGAE (GROUND)

%PLL SETTINGS

VCO_GAIN_R1 = (10.0001-LAGTIME)*1E3;                                        % DEFINE PLL R1 (~0-10K)               
VCO_GAIN_R2 = 0;                                                            % DEFINE PLL R2
target_mult = MULT_INTERVAL;                                                % DEFINE MULT COUNTER TARGET FROM USER INPUT
target_div = DIV_INTERVAL;                                                  % DEFINE DIV COUNTER TARGET FROM USER INPUT

% pll loopback filter       
R_response = (10.0001-RESPONSE)*200E3;                                      % DEFINE LOOP FILTER RESPONE RESISTANCE (0-2M)
R_looptrack = (LOOPTRACK)*50E3;                                             % DEFINE LOOP FILTER LOOPTRACK RESISTANCE (0-500K)
    
if LOOPSPEED == 3                                                           % DEFINE LOOP FILTER LOOPSPEED CAPACITANCE (.47uF, 2.67uF, 3.77uF)
    C_loopspeed = 33.47E-6;
end
if LOOPSPEED == 1
    C_loopspeed = .47E-6;
end
if LOOPSPEED == 2
    C_loopspeed = 2.67E-6;
end

% VOLUME SETTINGS                                                           % SET OUTPUT GAIN LEVELS
square_wave_volume = SQAREWAVE/10;
mult_volume = MULTIPLIER/10;
div_volume = DIVIDER/10;
R_pot_ws = WAVESHAPE*25E3;                                                  % WAVESHAPE FILTER RESISTANCE (0-250K)
master_vol = log10(MASTER);

% ------------------------------------------------------------------------

%% INPUT FILTERING PRE-PROCESSING

% COMPONENT VALUES -------------------------------------------------------
R0 = 10E3;
C0 = 10E-12;
R1 = 150E3;
C1 = 10E-6;
C3 = 10E-6;
C2 = 100E-12;
R3 = 1E3;
RGAIN = PREAMP*1000;

R1op = 200E3;
C1op = 220E-9;
R2op = 1E6;
C2op = 47E-12;
C3op = 100E-9;
R3op = 10E3;
R4op = 100E3;
% ------------------------------------------------------------------------

% PREAMP STAGE 1 ---------------------------------------------------------

A1 = - [0 -1 0 0; ...                                                       % STATE MATRIX FIRST PREAMP
       (1)/(C0*C1*R0*R1) , (C1*(R0+R1) + C0*R1)/(C0*C1*R0*R1), 0, 0; ...
       0, -((R1*C1)/(R3*C2)), 1/(RGAIN*C2), 1/(R3*C2); ...
       0, -((R1*C1)/(R3*C3)), 0, 1/(R3*C3)];

b1 = [0; ...                                                                % INPUT MATRIX FIRST PREAMP
     (1)/(C0*C1*R0*R1);...
     0;...
     0];

c1 = [0, R1*C1, 1, 0];                                                      % OUTPUT MATRIX FIRST PREAMP

I4 = eye(4);                                                                % IDENTITY MATRIX
bk1 = k * b1;                                                               % INPUT VECTOR * k
Bf1 = (I4 + k*A1/2);                                                        % I+kA/2 MATRIX
Bb1 = (I4 - k*A1/2);                                                        % I-kA/2 MATRIX
Bb1_inv = Bb1^(-1);                                                         % INVERTED I-kA/2 MATRIX
invCOEF1_1 = Bb1_inv*Bf1;                                                   % TRAPEZOID RULE MATRIX COEFFICIENT
invCOEF1_2 = Bb1_inv*bk1;                                                   % TRAPEZOID RULE MATRIX COEFFICIENT

xt1 = [0 0 0 0]';                                                           % PREALLOCATE STATE VECTOR
% ------------------------------------------------------------------------


% PREAMP STAGE 2 ---------------------------------------------------------

A2 = [-1/(R1op*C1op) 0 0; ...                                               % STATE MATRIX FIRST PREAMP
     1/(R1op*C2op) -1/(R2op*C2op) 0 ;...
     1/((R3op+R4op)*C3op) 1/((R3op+R4op)*C3op) -1/((R3op+R4op)*C3op)];

b2 = [1/(R1op*C1op); ...                                                    % INPUT MATRIX FIRST PREAMP
     -1/(R1op*C2op); ...
     1/((R3op+R4op)*C3op)];


c2 = [(R4op/((R3op+R4op))) ...                                              % OUTPUT MATRIX FIRST PREAMP
      (R4op/((R3op+R4op))) ...
      -R4op/((R3op+R4op))];                                  

I3 = eye(3);                                                                % IDENTITY MATRIX
bk2 = k * b2/2;                                                             % INPUT VECTOR * k
Bf2 = (I3 + k*A2/2);                                                        % I+kA/2 MATRIX
Bb2 = (I3 - k*A2/2);                                                        % I-kA/2 MATRIX
Bb2_inv = Bb2^(-1);                                                         % INVERTED I-kA/2 MATRIX
invCOEF2_1 = Bb2_inv*Bf2;                                                   % TRAPEZOID RULE MATRIX COEFFICIENT
invCOEF2_2 = Bb2_inv*bk2;                                                   % TRAPEZOID RULE MATRIX COEFFICIENT

xt2 = [0 0 0]';                                                             % PREALLOCATE STATE VECTOR
% ------------------------------------------------------------------------

% OUTPUT ALLOCATION
yt1 = 0;                                                                    % FILTER STAGE 1 OUTPUT - CURRENT SAMPLE
yt1prev = 0;                                                                % FILTER STAGE 1 OUTPUT - N-1 SAMPLE                          
filter_output = 0;                                                          % FILTER STAGE 2 OUTPUT - CURRENT SAMPLE

%% SATURATION STAGE PRE-PROCESSING

% OPEN LOOP AMP ----------------------------------------------------------

gain = 10^(AOL/20);                                                         % OPEN LOOP GAIN CALCULATION
y_ol = 0;                                                                   % OPEN LOOP OUTPUT - CURRENT SAMPLE
y_olprev = 0;                                                               % OPEN LOOP OUTPUT - N-1 SAMPLE
% ------------------------------------------------------------------------

% SCHMITT TRIGGER --------------------------------------------------------
state = 0;                                                                  % SCHMITT TRIGGER CURRENT STATE (HI/LO)
sat_out = 0;                                                                % SCHMITT TRIGGER OUTPUT - CURRENT SAMPLE
sat_outprev = 0;                                                            % SCHMITT TRIGGER OUTPUT - N-1 SAMPLE

%% PHASE LOCKED LOOP PRE-PROCESSING

% PLL INPUT --------------------------------------------------------------
x_in_pll = 0;                                                               % PLL INPUT - CURRENT SAMPLE
x_in_pllprev = 0;                                                           % PLL INPUT - N-1 SAMPLE
x_in_pllprevprev = 0;                                                       % PLL INPUT - N-2 SAMPLE
% ------------------------------------------------------------------------

% PHASE FREQUENCY DETECTOR -----------------------------------------------
QoutA = 0;                                                                  % PFD OUTPUT STATE A
QoutB = 0;                                                                  % PFD OUTPUT STATE B

x_ctrl = 0;                                                                 % PFD VOLTAGE OUTPUT - CURRENT SAMPLE
x_ctrlprev = 0;                                                             % PFD VOLTAGE OUTPUT - N-1 SAMPLE              
% ------------------------------------------------------------------------

% LOOP FILTER ------------------------------------------------------------
A_LF = -(1)/((R_response+R_looptrack)*C_loopspeed);                         % STATE MATRIX LOOP FILTER
B_LF = (k*.5)/((R_response+R_looptrack)*C_loopspeed);                          % INPUT MATRIX LOOP FILTER
C_LF = 1 - ((R_looptrack)/(R_response+R_looptrack));                        % OUTPUT MATRIX LOOP FILTER
D_LF = (R_looptrack)/((R_response+R_looptrack));                            % FEEDTHROUGH MATRIX LOOP FILTER

A_LF_OPEN = -(1)/((R_looptrack)*C_loopspeed);                                % STATE MATRIX LOOP FILTER - HIGH IMPEDANCE

Ak_LF_b = (1-k*A_LF/2);                                                     % I-kA/2 MATRIX
Ak_LF_b_inv = (1-k*A_LF/2)^(-1);
Ak_LF_f = (1+k*A_LF/2);                                                     % I+kA/2 MATRIX
Ak_LF_b_OPEN = (1-k*A_LF/2);                                                % I-kA/2 MATRIX
Ak_LF_b_OPEN_inv = (1-k*A_LF/2)^(-1);
Ak_LF_f_OPEN = (1+k*A_LF/2);                                                % I+kA/2 MATRIX

x_LF = 0;                                                                   % LOOP FILTER INPUT - CURRENT SAMPLE
x_LFprev = 0;                                                               % LOOP FILTER INPUT - N-1 SAMPLE

y_LF = 0;                                                                   % LOOP FILTER OUTPUT - CURRENT SAMPLE
% ------------------------------------------------------------------------

% VOLTAGE CONTROLLED OSCILLATOR ------------------------------------------
Fctrl = 0;                                                                  % VCO FREQUENCY
Fgain = (6.2/VCO_GAIN_R1)/(2*(.02E-6)*1.8);                                 % VCO FREQUENCY GAIN COEFFICIENT
% ------------------------------------------------------------------------

% DECADE COUNTER ---------------------------------------------------------
counter_pulse_mult = floor((.15E-3)*SR);                                    % FIXED LENGTH MULTIPLIER COUNTER PULSE 
mult_pulse_sample = 0;                                                      % PULSE TIMER

y_osc = 0;                                                                  % VCO OUTPUT - CURRENT SAMPLE
y_oscprev = 0;                                                              % VCO OUTPUT - N-1 SAMPLE

y_oscref = 0;                                                               % MULT COUNTER OUTPUT (DIVIDED VCO OUT) - CURRENT SAMPLE
y_oscrefprev = 0;                                                           % MULT COUNTER OUTPUT (DIVIDED VCO OUT) - N-1 SAMPLE

y_oscdiv = 0;                                                               % DIV COUNTER OUTPUT (DIVIDED MULT SIGNAL) - CURRENT SAMPLE

y_oscmult = 0;                                                              % MULT SIGNAL - CURRENT SAMPLE
    
y_oscdiv_in = 0;                                                            % DIVIDER INPUT SIGNAL - CURRENT SAMPLE
y_oscdiv_inprev = 0;                                                        % DIVIDER INPUT SIGNAL - N-1 SAMPLE

state_mult = 0;                                                             % MULT COUNTER STATE
state_div = 0;                                                              % DIV COUNTER STATE
counter_mult = 0;                                                           % MULT COUNTER RISING EDGE COUNT
counter_div = 0;                                                            % DIV COUNTER RISING EDGE COUNT
trig_mult = 0;                                                              % MULT COUNTER TRIGGER
trig_div = 0;                                                               % DIV COUNTER TRIGGER
% ------------------------------------------------------------------------

%% OUTPUT 

output = 0;                                                                 % PRE WAVESHAPE SUMMED OUTPUT

% OUTPUT PASSIVE FILTERING -----------------------------------------------
% COMPONENT VALUES
R1_ws = 10E3;
C1_ws = 10E-9;

A_ws = -(1)/((R1_ws+R_pot_ws)*C1_ws);                                       % STATE MATRIX WAVESHAPE FILTER
B_ws = (1)/((R1_ws+R_pot_ws)*C1_ws);                                        % INPUT MATRIX WAVESHAPE FILTER
C_ws = 1 - ((R_pot_ws)/(R1_ws+R_pot_ws));                                   % OUTPUT MATRIX WAVESHAPE FILTER
D_ws = (R_pot_ws)/((R1_ws+R_pot_ws));                                       % FEEDTHROUGH MATRIX WAVESHAPE FILTER


BK_ws = (k*B_ws/2);                                                         % K*INPUT MATRIX                                                                                                   
Ak_ws_b = (1-k*A_ws/2);                                                     % I-kA/2 MATRIX
Ak_ws_f = (1+k*A_ws/2);                                                     % I+kA/2 MATRIX

x_ws = 0;                                                                   % WAVESHAPE FILTER INPUT - CURRENT SAMPLE
output_prev = 0;                                                            % WAVESHAPE FILTER INPUT - N-1 SAMPLE
y_ws = 0;                                                                   % WAVESHAPE FILTER OUTPUT - CURRENT SAMPLE
% ------------------------------------------------------------------------      

master_out = zeros(Nf,1);                                                   % MASTER OUTPUT VECTOR ALLOCATION

if ANALYSIS_MODE == 2                                                       % PREALLOCATE ANALYSIS VECTORS            
    LFOUT_MTX = zeros(Nf,1);
    OSCDIV_MTX = zeros(Nf,1);
    OSCMULT_MTX = zeros(Nf,1);
    SQWV_MTX = zeros(Nf,1);
end

RUNTIME_PRE = toc;                                                          % END PREPROCESSING TIMER

%% MAIN LOOP -------------------------------------------------------------

tic
for n = 2 : Nf
% INPUT FILTERING --------------------------------------------------------

    xt1 = invCOEF1_1*xt1 + invCOEF1_2*(x_in(n-1)+x_in(n));                  % UPDATE STATE xt1 FROM N TO N+1, TRAPEZOIDAL INTEGRATION
    yt1 = c1*xt1;                                                           % WRITE SAMPLE TO OUTPUT VECTOR yt1
    
    xt2 = invCOEF2_1*xt2 + invCOEF2_2*(yt1prev+yt1);                        % UPDATE STATE xt2 FROM N TO N+1, TRAPEZOIDAL INTEGRATION
    filter_output = c2*xt2;                                                 % WRITE SAMPLE TO OUTPUT VECTOR filter_output

    yt1prev = yt1;                                                          % UPDATE SAMPLE
% ------------------------------------------------------------------------

% SATURATION -------------------------------------------------------------

% OPEN LOOP OP AMP
    y_ol = (filter_output-BIAS) * gain;                                     % APPLY TRIGGER OFFSET AND OPEN LOOP GAIN
    
    if y_ol > SWEEP_HI_ol                                                   % HARD CLIP OUTPUT (SCHMITT CLIPPING STAGE FOLLOWS SO NO NEED TO APPLY ANTI-ALIASING)
        y_ol = SWEEP_HI_ol;
    end
    
    if y_ol < SWEEP_LO_ol
        y_ol = SWEEP_LO_ol;
    end

% SCHMITT TRIGGER

    if y_ol >= V_HI && y_olprev < V_HI                                      % PASSING VOLTAGE HIGH THRESHOLD SENDS OUTPUT TO ZERO (INVERTED)
        state = SWEEP_LO_st;
    end
    if y_ol <= V_LO && y_olprev > V_LO                                      % PASSING VOLTAGE LOW THRESHOLD SENDS OUTPUT TO ONE (INVERTED)
        state = SWEEP_HI_st;
    end

    y_olprev = y_ol;                                                        % UPDATE N-1 SAMPLE

% SWITCHABLE INVERTER STEP                                                  % IF MULT_PHASE IS 1 INVERT SIGNAL (SEE SCHEM)
    if MULT_PHASE == 0
        sat_out = state;
    else
        sat_out = 1-state;
    end
% ------------------------------------------------------------------------

% PHASE FREQUENCY DETECTOR -----------------------------------------------

    x_in_pll = sat_out;                                                     % REDEFINE sat_out FOR PLL PROCESSING
    
    if x_in_pllprev > .1 && x_in_pllprevprev <.1                            % INPUT RISING EDGE DETECTOR (D FLIP FLOP A)
        QoutA = 1;
    end
    
    if y_oscref > .1 && y_oscrefprev <.1                                    % REFERENCE RISING EDGE DETECTOR (D FLIP FLOP B)
        QoutB = 1;
    end

    x_in_pllprevprev = x_in_pllprev;                                        % UPDATE REFERENCE SAMPLES
    x_in_pllprev = x_in_pll;                                                % UPDATE REFERENCE SAMPLES
    y_oscrefprev = y_oscref;                                                % UPDATE REFERENCE SAMPLES
    
    if QoutB && QoutA == 1                                                  % D FLIP FLOP AND GATE RESET
        QoutB = 0;
        QoutA = 0;
    end  
    
    if QoutB == 1                                                           % CHARGE PUMP OUTPUT LOW
        x_ctrl = 0;
    end
    
    if QoutA ==1                                                            % CHARGE PUMP OUTPUT HIGH
        x_ctrl = 11.85;
    end

    
    if QoutA ==1 || QoutB ==1                                               % STATE SPACE FOR OPEN CHARGE PUMP
        x_LF = Ak_LF_b_inv * ((x_ctrl+x_ctrlprev)*B_LF + Ak_LF_f*x_LFprev); % STATE UPDATE
        y_LF = C_LF*x_LF + D_LF*x_ctrl;                                     % LF OUTPUT
    else                                                                    % STATE SPACE FOR CLOSED CHARGE PUMP (HI Z)
        x_ctrl = 5.925;                                                     % PLACEHOLDER VCTRL (HI Z)
        x_LF = Ak_LF_b_OPEN_inv * (Ak_LF_f_OPEN*x_LFprev);                  % STATE UPDATE (HI Z)
        y_LF = x_LF;                                                        % LF OUTPUT (HI Z)
    end
    
    x_ctrlprev=x_ctrl;                                                      % SAMPLE UPDATE
    x_LFprev = x_LF;                                                        % SAMPLE UPDATE
% ------------------------------------------------------------------------

% VOLTAGE CONTROLLED OSCILLATOR ------------------------------------------
    t1=(n)*k;                                                               % TIME INDEX
    
    Fctrl = y_LF*(LAGTIME)*100;                                             % FREQUENCY GAIN CONTROL
    %Fctrl = y_LF*(LAGTIME)*Fgain;                                          % ALTERNATE FREQUENCY GAIN CONTROL
    y_osc = sin(2*pi*Fctrl*t1);                                             % SINE WAVE VCO GENERATION
    
    if y_osc > 0                                                            % SINE -> SQUARE WAVE VCO CLIPPING
        y_osc = 1;
    end
    
    if y_osc < 0
        y_osc = 0;
    end
% ------------------------------------------------------------------------


% DECADE COUNTERS --------------------------------------------------------
    % MULTIPLIER


    if y_osc > y_oscprev                                                    % RISING EDGE DETECTION
        counter_mult = counter_mult+1;                                      % UPDATE EDGE COUNTER
          
        if counter_mult == target_mult                                      % WHEN COUNTER EQUALS TARGET SET STATE OUTPUT TO HIGH
            state_mult = 1;
            counter_mult = 0;                                               % RESET EDGE COUNTER
            trig_mult = 1;                                                  % TRIGGER RESET ON NEXT RISING EDGE
        end
    end
    
    if trig_mult == 1                                                       % FIXED DURATION PULSE TIMER
        mult_pulse_sample = mult_pulse_sample+1;
        if mult_pulse_sample == counter_pulse_mult
            mult_pulse_sample = 0;
            trig_mult = 0;
            state_mult = 0;
        end
    end

    y_oscprev = y_osc;                                                      % SAMPLE UPDATE
    y_oscref = state_mult;                                                  % RECORD OUTPUT (PLL COMP IN)
    
    y_oscmult = 1-y_osc;                                                    % INVERT SIGNAL (MULTIPLIER OUTPUT)
    
    if DIV_PHASE == 1                                                       % IF NEGATIVE PHASE INVERT MULT PHASE
        y_oscdiv_in = 1-y_oscmult;
    else
        y_oscdiv_in = y_oscmult;
    end

    % DIVIDER
    if y_oscdiv_in > y_oscdiv_inprev                                        % RISING EDGE DETECTION             
        counter_div = counter_div+1;                                        % UPDATE EDGE COUNTER
        
        if trig_div == 1                                                    % RETURN HIGH STATE TO LOW ON FOLLOWING EDGE
            trig_div = 0;
            state_div = 0;
        end
        
        if counter_div == 1                                                 % ON COUNTER = 1 RAISE STATE (SEE SECOND COUNTER CONFIGURATION)
           state_div = 1;
           trig_div = 1;                                                    % TRIGGER RESET ON NEXT RISING EDGE
        end
    
        if counter_div == target_div                                        % RESET COUNTER AT TARGET
            counter_div = 0;
        end
        
    end
    
    y_oscdiv_inprev = y_oscdiv_in;                                          % SAMPLE UPDATE
    
    y_oscdiv = 1-state_div;                                                 % RECORD COUNTER STATE - INVERT

    
% ------------------------------------------------------------------------

% OUTPUT FILTERING -------------------------------------------------------
    output = y_oscmult*1.185*mult_volume+y_oscdiv*1.185*div_volume...
        +(y_ol)*square_wave_volume*.1;                                      % SUM OUTPUT SIGNALS Account for output voltages

    if output > SWEEP_HI_ol
        output = SWEEP_HI_ol;
    else 
        if output < 0
            output = 0;
        end
    end

    x_ws = Ak_ws_b \ (Ak_ws_f*x_ws + BK_ws*(output_prev+output));           % TRAPEZOID STATE SPACE UPDATE                          
    y_ws = C_ws*x_ws + D_ws*output;                                         % OUTPUT UPDATE
    
    output_prev = output;
% ------------------------------------------------------------------------
    master_out(n) = y_ws*master_vol*.1;                                     % MASTER VOLUME

    if ANALYSIS_MODE == 2                                                   % RECORD TO ANALYSIS VECTORS
        LFOUT_MTX(n) = y_LF;
        OSCDIV_MTX(n) = y_oscdiv;
        OSCMULT_MTX(n) = y_oscmult;
        SQWV_MTX(n) = y_ol;
        OSCREF_MTX(n) = y_oscref;
    end

end

RUNTIME_MAIN = toc;                                                         % STOP MAIN PROCESSING TIMER


% RUNTIME ANALYSIS
disp(['DATA PRE-PROCESSING COMPLETED IN ',num2str(RUNTIME_PRE),' SECONDS'])
disp(['MAIN BLOCK PROCESSING COMPLETED IN ',num2str(RUNTIME_MAIN),' SECONDS'])
disp(['TOTAL RUNTIME: ',num2str(RUNTIME_MAIN+RUNTIME_PRE),' SECONDS'])
disp(['PROCESSING SPEED IS ', num2str(round(((Nf/SR)/(RUNTIME_MAIN+RUNTIME_PRE)),4)),' times FASTER THAN AUDIO RATE, WITH AN EXCESS OF ', num2str((Nf/SR)-(RUNTIME_MAIN+RUNTIME_PRE)),' SECONDS'])

HSIG = fft(master_out) ;                                                    % TAKE FFT OF OUTPUT
HIN = fft(x_in(2:end));                                                     % TAKE FFT OF INPUT

figure                                                                      % DISPLAY FFT OF OUTPUT SIGNAL
loglog(fvec, abs(HSIG), 'b', fvec, abs(HIN),'r')
legend('Output','Input')
title("FFT of Input and Output Audio Signals", 'Interpreter', 'latex')
xlabel("Frequency [Hz]", 'Interpreter', 'latex')
ylabel("Amplitude [V]", 'Interpreter', 'latex')
xlim([20 20E3])
if INPUT_MODE == 2
    xline(f0,'-.','IN FREQ', 'HandleVisibility', 'off')
    xline(f0*MULT_INTERVAL,'-.','MULT FREQ', 'HandleVisibility', 'off')
    xline(f0*MULT_INTERVAL/DIV_INTERVAL,'-.','DIV FREQ', 'HandleVisibility', 'off')
    legend('Output','Input')
end

if SOUNDOUT == 1
soundsc(master_out,SR)                                                      % PLAY OUTPUT SOUND    
end

audiowrite(OUTPUT_FILENAME,...
    master_out-((max(master_out)-min(master_out))/2),SR)                     % WRITE OUTPUT AUDIO FILE

% FILTER FREQUENCY RESPONSES ---------------------------------------------
if ANALYSIS_MODE == 2
    yt_pre = zeros(Nf, 1);                                                  % PREALLOCATE OUTPUT
    yt_mid = zeros(Nf, 1);
    yt_loop = zeros(Nf, 1);
    Hc_loop = zeros(Nf, 1);
    u_test = [1 ; zeros(Nf-1, 1)];                                          % GENERATE INPUT DELTA FUNCTION
    xt1_TEST = [0 0 0 0]';                                                  % DEFINE STATE VECTORS
    xt2_TEST = [0 0 0]';
    xt3_TEST = 0;
    
    
    for n = 2 : Nf                                                          % COMPUTE FREQUENCY RESPONSES
    xt1_TEST = Bb1 \ (Bf1*xt1_TEST + bk1*(u_test(n-1)+u_test(n)));             
    yt_pre(n) = c1*xt1_TEST;                                     
    xt2_TEST = Bb2 \ (Bf2*xt2_TEST + bk2*(yt_pre(n-1)+yt_pre(n)));
    yt_mid(n) = c2*xt2_TEST;
    xt3_TEST = Ak_LF_b \ (Ak_LF_f*xt3_TEST + B_LF*(u_test(n-1)+u_test(n)));
    yt_loop(n) = C_LF*xt3_TEST + D_LF*u_test(n);
    Hc_loop(n) = C_LF*((2*pi*j*fvec(n)).*1-A_LF)^(-1)*(B_LF/(k));
    end


% INPUT FILTER FREQ ANALYSIS
figure
loglog(fvec, abs(fft(yt_pre)),'r',fvec, abs(fft(yt_mid)),'b','LineWidth',2)
legend('STAGE 1','STAGE 1+2')
grid on
xlim([10 20E3])
xlabel('Frequency [Hz]', 'Interpreter', 'latex')
ylabel('$|H(2 \pi j f)|$', 'Interpreter', 'latex')
title('Transfer Function of Schumann PLL Input Preamp Filtering', 'Interpreter', 'latex')

% LOOP FILTER FREQ ANALYSIS
figure
subplot(2,1,1)
loglog(fvec, abs(fft(yt_loop)+D_ws),'b',fvec, abs(Hc_loop+D_ws),'r')
grid on
legend('Trapezoid Rule','Analytical Solution')
xlim([10 20E3])
xlabel('Frequency [Hz]', 'Interpreter', 'latex')
ylabel('$|H(2 \pi j f)|$', 'Interpreter', 'latex')
title('Transfer Function of Schumann PLL Loop Filtering', 'Interpreter', 'latex')

% LOOP FILTER OUTPUT
subplot(2,1,2)
plot(t,LFOUT_MTX, 'b');
grid on
title("Loop Filter Output", 'Interpreter', 'latex')
ylabel("Amplitude [V]", 'Interpreter', 'latex')
xlabel("Time [s]", 'Interpreter', 'latex')

% DECOMPOSED OUTPUT FREQ ANALYSIS
if INPUT_MODE == 2
figure
loglog(fvec, abs(fft(SQWV_MTX)), 'g', fvec, abs(fft(OSCMULT_MTX)), 'r', fvec, abs(fft(OSCDIV_MTX)), 'B')
legend('SQUARE WAVE', 'MULTIPLIER', 'DIVIDER')
grid on
title("Frequency Response of Signal Components", 'Interpreter', 'latex')
xline(f0,'-.','Input Frequency', 'HandleVisibility', 'off')
xline(f0*MULT_INTERVAL,'-.','Multiplier Frequency', 'HandleVisibility', 'off')
xline(f0*MULT_INTERVAL/DIV_INTERVAL,'-.','Divider Frequency','HandleVisibility', 'off')
xlabel('Frequency [Hz]', 'Interpreter', 'latex')
ylabel('$|H(2 \pi j f)|$', 'Interpreter', 'latex')
xlim([20 20E3])

figure
loglog(fvec, abs(fft(SQWV_MTX)), 'r', fvec, abs(fft(OSCREF_MTX)), 'b')
legend('SIG IN', 'COMP IN')
grid on
title("4046 PLL Frequency Locking", 'Interpreter', 'latex')
xline(f0,'--','DisplayName','Input Frequency')
xlabel('Frequency [Hz]', 'Interpreter', 'latex')
ylabel('$|H(2 \pi j f)|$', 'Interpreter', 'latex')
xlim([20 20E3])

else
figure
loglog(fvec, abs(fft(SQWV_MTX)), 'g', fvec, abs(fft(OSCMULT_MTX)), 'r', fvec, abs(fft(OSCDIV_MTX)), 'b')
legend('SQUARE WAVE', 'MULTIPLIER', 'DIVIDER')
grid on
title("frequency response of signal components", 'Interpreter', 'latex')
xlabel('Frequency [Hz]', 'Interpreter', 'latex')
ylabel('$|H(2 \pi j f)|$', 'Interpreter', 'latex')
xlim([20 20E3])
end

end
% ------------------------------------------------------------------------
