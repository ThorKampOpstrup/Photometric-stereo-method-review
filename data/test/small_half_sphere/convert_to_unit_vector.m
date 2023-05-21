lights = readmatrix("light_positions copy.txt")


subtract =  0.1

before = lights(:,3);
lights(:,3) = lights(:,3)-subtract;

%normalice the light positions to univ vectors
for i=1:size(lights,1)
    lights(i,:) = lights(i,:)/norm(lights(i,:));
end

plot3(lights(:,1),lights(:,2),lights(:,3),'o')
lights

writematrix(lights, "light_positions")