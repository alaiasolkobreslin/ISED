nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

multi_addition(W,X,Y,Z) :- number(W,W2),number(X,X2),number(Y,Y2), Z is W2+X2+Y2.
addition(W,X,Y,Z) :- digit(W,W2), digit(X,X2), digit(Y,Y2), Z is W2+X2+Y2.