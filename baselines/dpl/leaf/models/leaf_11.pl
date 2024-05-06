nn(leaf_net_margin,[X],Y,['serrate','dentate','lobed','serrulate','entire','undulate']) :: margin(X,Y).
nn(leaf_net_shape,[X],Y,['ovate','lanceolate','oblong','obovate','elliptical']) :: shape(X,Y).
nn(leaf_net_texture,[X],Y,['leathery', 'smooth', 'glossy', 'medium']) :: texture(X,Y).

leaf_11('serrate',   _,            _,         4). % Ocimum basilicum
leaf_11('dentate',   _,            _,         2). % Jatropha curcas
leaf_11('lobed',     _,            _,         5). % Platanus orientalis
leaf_11('serrulate', _,            _,         1). % Citrus limon
leaf_11('entire',   'ovate',       _,         6). % Pongamia Pinnata
leaf_11('entire',   'lanceolate',  _,         3). % Mangifera indica
leaf_11('entire',   'oblong',      _,         9). % Syzygium cumini
leaf_11('entire',   'obovate',     _,         7). % Psidium guajava
leaf_11('entire',   'elliptical', 'leathery', 0). % Alstonia Scholar
leaf_11('entire',   'elliptical', 'smooth',   10). % Terminalia Arjuna
leaf_11('entire',   'elliptical', 'glossy',   1). % Citrus limon
leaf_11('entire',   'elliptical', 'medium',   8). % Punica granatum
leaf_11('undulate', 'ovate',       _,         9). % Syzygium cumini
leaf_11('undulate', 'lanceolate',  _,         3). % Mangifera indica
leaf_11('undulate', 'oblong',      _,         9). % Syzygium cumini
leaf_11('undulate', 'obovate',     _,         9). % Syzygium cumini
leaf_11('undulate', 'elliptical',  _,         10). % Terminalia Arjuna


main(X, L) :- margin(X, M2), shape(X, S2), texture(X, T2),
                    leaf_11(M2, S2, T2, L).
