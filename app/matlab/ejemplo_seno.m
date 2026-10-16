% EJEMPLO_SENO
% Programa básico de MATLAB: genera y grafica una señal senoidal.

clear;
clc;
close all;

% Parámetros de la señal
amplitud = 2;
frecuencia = 5; % Hz
fase = pi / 4;  % radianes
duracion = 1;   % segundos

% Vector de tiempo con 1000 muestras
tiempo = linspace(0, duracion, 1000);

% Ecuación de la señal: y(t) = A * sin(2*pi*f*t + fase)
senal = amplitud * sin(2 * pi * frecuencia * tiempo + fase);

% Mostrar información básica en la ventana de comandos
fprintf('Amplitud: %.2f\n', amplitud);
fprintf('Frecuencia: %.2f Hz\n', frecuencia);
fprintf('Valor máximo calculado: %.2f\n', max(senal));
fprintf('Valor mínimo calculado: %.2f\n', min(senal));

% Crear la gráfica
figure('Name', 'Ejemplo de señal senoidal');
plot(tiempo, senal, 'b', 'LineWidth', 1.5);
grid on;

title('Señal senoidal');
xlabel('Tiempo (s)');
ylabel('Amplitud');
legend('y(t) = A sin(2\pi f t + fase)', 'Location', 'best');
