clear;
clc;
path = 'half_sphere_3pi_d8/log_file.csv'
plot_name = 'plate'

%read the log file
M = readmatrix(path)

n = M(:,1);
mean = M(:,2);
mean_abs = M(:,3);
sum = M(:,6);
std = M(:,7);
time = M(:,10);

%plot the mean
figure(1)
plot(n, mean)
title('Mean angular difference with golden standard(plate)')
xlabel('Light sources [n]')
ylabel('Mean [degrees]')

%plot the mean absolute
%figure(2)
%plot(n, mean_abs)
%title('Mean absolute angular difference with golden standard')
%xlabel('Light sources')
%ylabel('Mean absolute [degrees]')


%plot the sum
%figure(2)
%plot(n, sum)
%title('Sum angular difference with golden standard(plate)')
%xlabel('Light sources [n]')
%ylabel('Sum [degrees]')

%plot the std
%figure(4)
%plot(n, std)
%title('Standard deviation angular difference with golden standard')
%xlabel('Light sources')
%ylabel('Standard deviation [degrees]')

%plot the time
figure(2)
plot(n, time)
title('Time to compute normal map estimation(plate)')
xlabel('Light sources [n]')
ylabel('Time [s]')


