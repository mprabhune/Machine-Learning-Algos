function linear_regression(filename, degree, lamda)
d = load(filename, "-ascii");

x = d(:,1);
t = d(:,2);
plot (x, t);
grid on;
hold on;

if (degree == 1)
	p = [ones(size(x,1),1)'; x'];
	phi = p';
	#w = inv(phi' * phi) * phi' * t 
	w = inv((lamda * eye(2)) + phi' * phi) * phi' * t;
	tp = w(1) + (w(2) * x);
	plot (x, tp, 'r');
	printf("w0=%.4f\n", w(1));
	printf("w1=%.4f\n", w(2));
	printf("w2=%.4f\n", 0.0);
else
	#2nd deg poly
	p2 = [ones(size(x,1),1)'; x';x'.*x'];
	p2 = p2';
	w = inv((lamda * eye(3)) + p2' * p2) * p2' * t;
	tp2 = w(1) + (w(2) * x) + (w(3) * x .* x);
	plot(x,tp2,'g');
	printf("w0=%.4f\n", w(1));
	printf("w1=%.4f\n", w(2));
	printf("w2=%.4f\n", w(3));

end
end




