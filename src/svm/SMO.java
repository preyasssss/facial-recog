package svm;

import java.io.Serializable;

/**
 * Implementare proprie a algoritmului Sequential Minimal Optimization (SMO)
 * pentru antrenarea unei masini cu vectori suport (SVM) binare.
 * <p>
 * Algoritmul rezolva problema duala a SVM cu soft margin (parametrul C) si
 * nucleu generic (prin metoda {@link #kernel(double[], double[])}). In mod
 * implicit se foloseste nucleul Sigmoid: {@code tanh(gamma * dot(x,z) + coeff0)}.
 * <p>
 * Imbunatatiri fata de versiunea initiala:
 * <ul>
 *   <li>{@code maxPasses} a fost marit de la 5 la 10, deoarece pe seturi mici
 *       de date cu nucleu sigmoid SMO nu convergea suficient in doar 5 treceri.
 *       La 10 treceri convergenta este mult mai stabila.</li>
 * </ul>
 * <p>
 * Clasa este serializabila, astfel incat clasificatorii antrenati pot fi
 * salvati si incarcati ulterior.
 */
public class SMO implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Parametrul de regularizare (soft margin). */
    private double C;

    /** Parametrul gamma al nucleului Sigmoid. */
    private double gamma;

    /** Parametrul coeff0 al nucleului Sigmoid. */
    private double coeff0;

    /** Multiplicatorii Lagrange. */
    private double[] alphas;

    /** Bias-ul (termenul liber) al hiperplanului. */
    private double b;

    /** Matricea de caracteristici de antrenament. */
    private double[][] X;

    /** Etichetele exemplelor de antrenament (+1 sau -1). */
    private int[] y;

    /** Toleranta pentru conditiile KKT. */
    private double tolerance = 1e-3;

    /**
     * Numarul maxim de treceri fara modificari inainte de oprire.
     * <p>
     * FIX: era 5 — prea putine treceri, SMO nu converge pe seturi mici
     * de date cu kernel sigmoid. La 10 trece convergenta e mult mai stabila.
     */
    private int maxPasses = 10;

    /**
     * Constructor cu parametrii nucleului Sigmoid.
     *
     * @param C      parametrul de regularizare
     * @param gamma  parametrul gamma pentru tanh
     * @param coeff0 parametrul coeff0 pentru tanh
     */
    public SMO(double C, double gamma, double coeff0) {
        this.C = C;
        this.gamma = gamma;
        this.coeff0 = coeff0;
    }

    /**
     * Obtine valoarea bias-ului dupa antrenament (util pentru diagnostic).
     * @return bias-ul b
     */
    public double getB() {
        return b;
    }

    /**
     * Antreneaza clasificatorul SVM folosind algoritmul SMO.
     * <p>
     * Algoritmul simplificat:
     * <ol>
     *   <li>Initializeaza multiplicatorii Lagrange cu zero si bias-ul cu zero.</li>
     *   <li>Repeta pana la maximul de treceri fara schimbari:
     *       <ul>
     *           <li>Pentru fiecare exemplu i, daca incalca conditiile KKT,
     *               selecteaza aleator un al doilea exemplu j.</li>
     *           <li>Actualizeaza perechea (alpha_i, alpha_j) in limitele [0, C]
     *               si ajusteaza bias-ul.</li>
     *       </ul>
     *   </li>
     * </ol>
     *
     * @param features matricea de caracteristici (fiecare rand este un exemplu)
     * @param labels   etichetele corespunzatoare (+1 sau -1)
     */
    public void train(double[][] features, int[] labels) {
        this.X = features;
        this.y = labels;
        int N = X.length;
        alphas = new double[N];
        b = 0;
        int passes = 0;
        while (passes < maxPasses) {
            int numChanged = 0;
            for (int i = 0; i < N; i++) {
                double Ei = error(i);
                if ((y[i] * Ei < -tolerance && alphas[i] < C) ||
                    (y[i] * Ei > tolerance && alphas[i] > 0)) {
                    int j = selectSecond(i, N);
                    double Ej = error(j);
                    double oldAi = alphas[i], oldAj = alphas[j];
                    double L, H;
                    if (y[i] != y[j]) {
                        L = Math.max(0, alphas[j] - alphas[i]);
                        H = Math.min(C, C + alphas[j] - alphas[i]);
                    } else {
                        L = Math.max(0, alphas[i] + alphas[j] - C);
                        H = Math.min(C, alphas[i] + alphas[j]);
                    }
                    if (L == H) continue;
                    double eta = 2 * kernel(i, j) - kernel(i, i) - kernel(j, j);
                    if (eta >= 0) continue;
                    alphas[j] = oldAj - y[j] * (Ei - Ej) / eta;
                    alphas[j] = Math.min(H, Math.max(L, alphas[j]));
                    if (Math.abs(alphas[j] - oldAj) < 1e-5) continue;
                    alphas[i] = oldAi + y[i] * y[j] * (oldAj - alphas[j]);
                    double b1 = b - Ei - y[i] * (alphas[i] - oldAi) * kernel(i, i)
                                     - y[j] * (alphas[j] - oldAj) * kernel(i, j);
                    double b2 = b - Ej - y[i] * (alphas[i] - oldAi) * kernel(i, j)
                                     - y[j] * (alphas[j] - oldAj) * kernel(j, j);
                    if (alphas[i] > 0 && alphas[i] < C)      b = b1;
                    else if (alphas[j] > 0 && alphas[j] < C) b = b2;
                    else                                       b = (b1 + b2) / 2.0;
                    numChanged++;
                }
            }
            if (numChanged == 0) passes++;
            else passes = 0;
        }
    }

    /**
     * Calculeaza eroarea pentru exemplul i (predictia minus eticheta reala).
     */
    private double error(int idx) {
        return predict(X[idx]) - y[idx];
    }

    /**
     * Alege aleator un al doilea exemplu, diferit de i.
     */
    private int selectSecond(int i, int N) {
        int j = i;
        while (j == i) j = (int)(Math.random() * N);
        return j;
    }

    /**
     * Calculeaza kernelul intre doua exemple date prin indicii lor.
     */
    private double kernel(int i, int j) {
        return kernel(X[i], X[j]);
    }

    /**
     * Functia nucleu: produs scalar trecut prin tanh (nucleu Sigmoid).
     * <p>
     * Formula: {@code K(x,z) = tanh(gamma * dot(x,z) + coeff0)}
     * <p>
     * Poate fi suprascrisa pentru a testa alte nuclee (liniar, RBF etc.).
     *
     * @param x1 primul vector
     * @param x2 al doilea vector
     * @return valoarea kernelului
     */
    public double kernel(double[] x1, double[] x2) {
        double dot = 0.0;
        for (int k = 0; k < x1.length; k++) dot += x1[k] * x2[k];
        return Math.tanh(gamma * dot + coeff0);
    }

    /**
     * Calculeaza scorul de decizie pentru un nou vector de caracteristici.
     * <p>
     * Formula: {@code f(x) = sum_{i} alpha_i * y_i * K(X_i, x) - b}
     * <p>
     * Semnul scorului indica clasa (pozitiv = +1, negativ = -1), dar pentru
     * recunoastere se foloseste scorul brut pentru a compara persoanele.
     *
     * @param x vectorul de caracteristici
     * @return scorul de decizie
     */
    public double predict(double[] x) {
        double sum = -b;
        for (int i = 0; i < X.length; i++) {
            if (alphas[i] != 0) {
                sum += alphas[i] * y[i] * kernel(X[i], x);
            }
        }
        return sum;
    }
}