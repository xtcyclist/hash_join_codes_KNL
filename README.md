# hash_join_codes_KNL
Vectorized implementations of hash join algorithms on Intel Xeon Phi (KNL)

# Generate input relations
```
make write
./write [#threads] [size of the outer relation] [size of the input relation] 
```

# Run hash joins
```
./npj [#threads] [size of the outer relation] [size of the input relation]
./phj [#threads] [size of the outer relation] [size of the input relation]
```
