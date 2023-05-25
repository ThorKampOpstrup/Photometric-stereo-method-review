clc;
clear;

plate = readmatrix('plate/log_file.csv')
plate_mask_size = 2657526
viking = readmatrix('viking/log_file.csv')
viking_mask_size = 1189644
half_sphere = readmatrix('half_sphere_3pi_d8/log_file.csv')
half_sphere_mask_size = 432612

n_index = 1
mean_index = 2
time_index = 10

figure(1)
plot(plate(:, n_index), plate(:, mean_index))
hold on
plot(viking(:, n_index), viking(:, mean_index))
plot(half_sphere(:, n_index), half_sphere(:, mean_index))
legend('Plate', 'Viking', 'Half sphere')
xlabel('ligth sources [n]')
ylabel('Mean [Deg]')
title('Mean angular difference with golden standard')
%set size
%set(gcf,'units','points','position',[0,0,500,400])

figure(2)
plot(plate(:, n_index), plate(:, time_index))
hold on
plot(viking(:, n_index), viking(:, time_index))
plot(half_sphere(:, n_index), half_sphere(:, time_index))
legend('Plate', 'Viking', 'Half sphere')
xlabel('Ligth sources [n]')
ylabel('Time [s/px]')
title('Compute time of normal map')
%set(gcf,'units','points','position',[0,0,500,400])

figure(3)
plot(plate(:, n_index), plate(:, time_index)./plate_mask_size)
hold on
plot(viking(:, n_index), viking(:, time_index)./viking_mask_size)
plot(half_sphere(:, n_index), half_sphere(:, time_index)./half_sphere_mask_size)
legend('Plate', 'Viking', 'Half sphere')
xlabel('Ligth sources [n]')
ylabel('Time [s/px]')
title('Compute time of normal map normalized')
%set(gcf,'units','points','position',[0,0,500,400])

hold off
