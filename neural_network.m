function neural_network(training_file, test_file, layers, units_per_layer, rounds)
#training_file = "./pendigits_training.txt";
#test_file = "./pendigits_test.txt";
#layers = 2;
#units_per_layer = 20;
#rounds = 50;

f = load(training_file, "-ascii");
D = size(f,2)-1;

x = f(:,1:D);
N = size(x, 1);
t_raw = f(:,D+1);
K = size(unique(t_raw),1);
t = bsxfun(@eq, t_raw(:),unique(t_raw)');
n_fac = max(max(x));
x = x ./ n_fac;
L = layers;
U = 1 + D + (L - 2) * (1 + units_per_layer) + K;


if (L == 2)
	w_size = (1 + D) * K;
elseif (L == 3)
	w_size = (1 + D) * units_per_layer + (1 + units_per_layer) * K;
else
	w_size = (1 + D) * units_per_layer + (1 + units_per_layer) * units_per_layer + (1 + units_per_layer) * K;
end

z = zeros(1, U); % output of all neurons
a = zeros(1, U); % sum of all neurons
d = zeros(1, U); % deltas

w = (rand(1, w_size) - 0.5) ./ 10.0;
eta = 1.0;

for r = 1:rounds
	if (L == 2)
		for n = 1:N
			z(1, 1:D+1) = [1.0 x(n, :)];
			zi =  z(1, 1:D+1);
			for i = 0:K-1 
				a((D+1+i) + 1) = w(i * (D+1) + 1: i * (D+1) + 1 + D) * zi';
				z((D+1+i) + 1) = 1 ./ (1 + e.^-a((D+1+i) + 1));
	
				zj = z((D+1+i) + 1);
				dj = (zj - t(n, i + 1)) * zj * (1-zj);
				d((D+1+i) + 1) = dj;
				w(i * (D+1) + 1: i * (D+1) + 1 + D) -= eta * dj * zi;
			end
		end	
	elseif (L == 3)
		for n = 1:N
			z(1, 1:D+1) = [1.0 x(n, :)];
			zi =  z(1, 1:D+1);
			for i = 0: units_per_layer - 1
				a((D+1+i) + 1) = w(i * (D+1) + 1: i * (D+1) + 1 + D) * zi';
				z((D+1+i) + 1) = 1 ./ (1 + e.^-a((D+1+i) + 1));
			end
			off_units = 1 + D + 1 + units_per_layer;
			off = units_per_layer * (D+1);
			zi = z(D+1+1:D+1+units_per_layer - 1 + 1);

			for i = 0:K-1 
				a((off_units + i) + 1) = w(off + i * (units_per_layer) + 1:off + i * (units_per_layer) + 1 + units_per_layer - 1) * zi';
				z((off_units + i) + 1) = 1 ./ (1 + e.^-a((off_units + i) + 1));
	
				zj = z((off_units + i) + 1);
				dj = (zj - t(n, i + 1)) * zj * (1-zj);
				d((off_units + i) + 1) = dj;
				w(off + i * (units_per_layer) + 1:off + i * (units_per_layer) + 1 + units_per_layer - 1) -= eta * dj * zi;
			end
			zi = z(1:D+1);
			for i = 0: units_per_layer - 1
				zu = d(0 + 1:0 + 1 + units_per_layer -1) * w(0+1:D+1:(D+1) * (units_per_layer))';
				zj = z((D+1+i) + 1);
				dj = zu * zj * (1-zj);
				d((D+1+i) + 1) = dj;
				w(i * (D+1) + 1:i * (D+1) + 1 + D) -= eta * dj * zi;
			end
		end		
	else
	
	end 
#	printf("r = %d\n", r); % bug remove
	eta = 0.98 * eta;
end

tst = load(test_file, "-ascii");
x_tst = tst(:,1:D);
N_tst = size(x_tst, 1);
t_tst_raw = tst(:,D+1);
t_tst = bsxfun(@eq, t_tst_raw(:),unique(t_raw)');
t_tst_size = size(t_tst,1);
x_tst = x_tst ./ n_fac;

classification_accuracy = 0.0;	

for n = 1:t_tst_size
	t_class = t_tst_raw(n);
	if (L == 2)
		z(1, 1:D+1) = [1.0 x_tst(n, :)];
		zi =  z(1, 1:D+1);
		for i = 0:K-1
			a((D+1+i) + 1) = w(i * (D+1) + 1: i * (D+1) + 1 + D) * zi';
	%		z((D+1+i) + 1) = sigmoid(a((D+1+i) + 1));
			z((D+1+i) + 1) = 1 ./ (1 + e.^-(a((D+1+i) + 1)));
		end
		zj = max(z((D+1) + 1: D+1+1 + K-1));
		p_class = 0;
		acc = 0;
		n_ties = 0;
		found = 0;
		for j = 0:K-1
			if (zj <= z((D+1+j) + 1))
				p_class = j;
				n_ties += 1;
				if (p_class == t_class)
					found = 1;
				end
			end
		end
		if (found != 0)
			if (n_ties != 0)
				acc = 1.0 / n_ties;
			else
				acc = 1.0;
			end	
		else
			acc = 0.0;
		end
		classification_accuracy += acc/t_tst_size;
	elseif (L == 3)
		z(1, 1:D+1) = [1.0 x_tst(n, :)];
		zi =  z(1, 1:D+1);
		for i = 0: units_per_layer - 1
			a((D+1+i) + 1) = w(i * (D+1) + 1: i * (D+1) + 1 + D) * zi';
			z((D+1+i) + 1) = 1 ./ (1 + e.^-a((D+1+i) + 1));
		end
		off_units = 1 + D + 1 + units_per_layer;
		off = units_per_layer * (D+1);
		zi = z(D+1+1:D+1+units_per_layer - 1 + 1);	

		for i = 0:K-1 
			a((off_units + i) + 1) = w(off + i * (units_per_layer) + 1:off + i * (units_per_layer) + 1 + units_per_layer - 1) * zi';
			z((off_units + i) + 1) = 1 ./ (1 + e.^-a((off_units + i) + 1));
		end
		zj = max(z((off_units) + 1: off_units+1 + K-1));
		p_class = 0;
		acc = 0;
		n_ties = 0;
		found = 0;
		for j = 0:K-1
			if (zj <= z((off_units+j) + 1))
				p_class = j;
				n_ties += 1;
				if (p_class == t_class)
					found = 1;
				end
			end

		end
		if (found != 0)
			if (n_ties != 0)
				acc = 1.0 / n_ties;
			else
				acc = 1.0;
			end	
		else
			acc = 0.0;
		end
		classification_accuracy += acc/t_tst_size;
	else
	end
	printf("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n",n-1, p_class, t_class, acc);

end % main for test
printf("classification accuracy=%6.4f\n", classification_accuracy);
end % function end

