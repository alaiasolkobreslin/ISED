nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

not_3_4(Y, Z) :-
    digit(Y,Y2),
    Y2 \= 3,
    Y2 \= 4,
    Z is 1.

not_3_4(Y, Z) :-
    digit(Y,Y2),
    (Y2 = 3; Y2 = 4),
    Z is 0.