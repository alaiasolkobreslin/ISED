nn(leaf_net_margin,[X],Y,['serrate','dentate','lobed','serrulate','entire','undulate']) :: margin(X,Y).
nn(leaf_net_shape,[X],Y,['ovate','lanceolate','oblong','obovate','elliptical']) :: shape(X,Y).
nn(leaf_net_texture,[X],Y,['leathery', 'smooth', 'glossy', 'medium']) :: texture(X,Y).

leaf_11('serrate',   _,            _,         L) :- L = 'Ocimum basilicum'.
leaf_11('dentate',   _,            _,         L) :- L = 'Jatropha curcas'.
leaf_11('lobed',     _,            _,         L) :- L = 'Platanus orientalis'.
leaf_11('serrulate', _,            _,         L) :- L = 'Citrus limon'.
leaf_11('entire',   'ovate',       _,         L) :- L = 'Pongamia Pinnata'.
leaf_11('entire',   'lanceolate',  _,         L) :- L = 'Mangifera indica'.
leaf_11('entire',   'oblong',      _,         L) :- L = 'Syzygium cumini'.
leaf_11('entire',   'obovate',     _,         L) :- L = 'Psidium guajava'.
leaf_11('entire',   'elliptical', 'leathery', L) :- L = 'Alstonia Scholar'.
leaf_11('entire',   'elliptical', 'smooth',   L) :- L = 'Terminalia Arjuna'.
leaf_11('entire',   'elliptical', 'glossy',   L) :- L = 'Citrus limon'.
leaf_11('entire',   'elliptical', 'medium',   L) :- L = 'Punica granatum'.
leaf_11('undulate', 'ovate',       _,         L) :- L = 'Syzygium cumini'.
leaf_11('undulate', 'lanceolate',  _,         L) :- L = 'Mangifera indica'.
leaf_11('undulate', 'oblong',      _,         L) :- L = 'Syzygium cumini'.
leaf_11('undulate', 'obovate',     _,         L) :- L = 'Syzygium cumini'.
leaf_11('undulate', 'elliptical',  _,         L) :- L = 'Terminalia Arjuna'.


main(X, L) :- margin(X, M2), shape(X, S2), texture(X, T2),
                    leaf_11(M2, S2, T2, L).
