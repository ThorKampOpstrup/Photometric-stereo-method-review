M = readmatrix('light_positions.txt')

% take first element of M
plot3(M(:,1),M(:,2),M(:,3),'o')