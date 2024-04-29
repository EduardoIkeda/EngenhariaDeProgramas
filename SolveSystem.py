def SolveSystem(n,TO,TC,TA):
    
    def Ainv(n):
        
        # Define the matrix A
        #  A = np.array([[a,b,c], [d,e,f], [g,h,i]])
        
        a = 2*n
        b = n
        c = 3*n+1

        d = n
        e = 2*n
        f = 2*n+1

        g = n
        h = n
        i = 2*n+1

        # Calculating the determinant of A
        det_A = a*(e*i - f*h) + b*(f*g - d*i) + c*(d*h - e*g)

        #  Define the matrix A inverse
        #  Ainv = np.array([[A,B,C], [D,E,F], [g,H,I]])
        
        # Calculating each element of the inverse matrix A^-1
        A = (e*i - f*h) / det_A
        B = (c*h - b*i) / det_A
        C = (b*f - c*e) / det_A

        D = (f*g - d*i) / det_A
        E = (a*i - c*g) / det_A
        F = (c*d - a*f) / det_A

        G = (d*h - e*g) / det_A
        H = (b*g - a*h) / det_A
        I = (a*e - b*d) / det_A
        
        return A,B,C,D,E,F,G,H,I

    A,B,C,D,E,F,G,H,I =  Ainv(n)

    # multiply Ainv by the experimental Time vector [TO,TC,TA]
    ta = A * TA + B * TC + C * TO
    tc = D * TA + E * TC + F * TO
    to = G * TA + H * TC + I * TO

    return to, tc, ta # tempos unitários das primitivas

