nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

count_threes_and_fours([], 0).  % Base case: an empty list has zero threes and fours
count_threes_and_fours([3|T], N) :- count_threes_and_fours(T, N1), N is N1 + 1.
count_threes_and_fours([4|T], N) :- count_threes_and_fours(T, N1), N is N1 + 1.
count_threes_and_fours([H|T], N) :- not(H is 3), not(H is 4),
                                    count_threes_and_fours(T, N).  % Ignore other numbers

main(List, Result) :- count_threes_and_fours(List, Result).

count_3_4(A,B,C,D,E,F,G,H,Z) :- digit(A,A2), digit(B,B2), digit(C,C2), digit(D,D2),
                                digit(E,E2), digit(F,F2), digit(G,G2), digit(H,H2),
                                main([A2, B2, C2, D2, E2, F2, G2, H2], Z).