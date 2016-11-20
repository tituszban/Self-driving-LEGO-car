function W = randInitialThetas(L_in, L_out)

epsilon_init = 0.12;

W = rand(L_out, L_in + 1) * (2 * epsilon_init) - epsilon_init;