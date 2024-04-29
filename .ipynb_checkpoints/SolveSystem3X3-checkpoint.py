{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "52d4cd9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def SolveSystem(n,TO,TC,TA):\n",
    "    \n",
    "    def Ainv(n):\n",
    "        \n",
    "        # Define the matrix A\n",
    "        #  A = np.array([[a,b,c], [d,e,f], [g,h,i]])\n",
    "        \n",
    "        a = 2*n\n",
    "        b = n\n",
    "        c = 3*n+1\n",
    "\n",
    "        d = n\n",
    "        e = 2*n\n",
    "        f = 2*n+1\n",
    "\n",
    "        g = n\n",
    "        h = n\n",
    "        i = 2*n+1\n",
    "\n",
    "        # Calculating the determinant of A\n",
    "        det_A = a*(e*i - f*h) + b*(f*g - d*i) + c*(d*h - e*g)\n",
    "\n",
    "        #  Define the matrix A inverse\n",
    "        #  Ainv = np.array([[A,B,C], [D,E,F], [g,H,I]])\n",
    "        \n",
    "        # Calculating each element of the inverse matrix A^-1\n",
    "        A = (e*i - f*h) / det_A\n",
    "        B = (c*h - b*i) / det_A\n",
    "        C = (b*f - c*e) / det_A\n",
    "\n",
    "        D = (f*g - d*i) / det_A\n",
    "        E = (a*i - c*g) / det_A\n",
    "        F = (c*d - a*f) / det_A\n",
    "\n",
    "        G = (d*h - e*g) / det_A\n",
    "        H = (b*g - a*h) / det_A\n",
    "        I = (a*e - b*d) / det_A\n",
    "        \n",
    "        return A,B,C,D,E,F,G,H,I\n",
    "\n",
    "    A,B,C,D,E,F,G,H,I =  Ainv(n)\n",
    "\n",
    "    # multiply Ainv by the experimental Time vector [TO,TC,TA]\n",
    "    ta = A * TA + B * TC + C * TO\n",
    "    tc = D * TA + E * TC + F * TO\n",
    "    to = G * TA + H * TC + i * TO\n",
    "\n",
    "    return to, tc, ta # tempos unitários das primitivas\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3b594549-b9d7-4c61-b187-ee0c805c831c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
