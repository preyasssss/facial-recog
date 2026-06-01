package recognition;

import svm.SMO;
import java.io.Serializable;
import java.util.List;

/**
 * Clasa model care stocheaza datele unei persoane pentru recunoasterea faciala.
 * <p>
 * Contine pseudonimul, clasificatorul SVM antrenat (one‑vs‑all) si
 * lista vectorilor HOG folositi la antrenament (utila pentru diagnostica).
 * <p>
 * Implementeaza {@link Serializable} pentru a putea fi salvata si incarcata
 * impreuna cu intregul sistem de recunoastere.
 */
public class PersonModel implements Serializable {

    /** Versiunea de serializare (pentru compatibilitate). */
    private static final long serialVersionUID = 1L;

    /**
     * Pseudonimul persoanei (folosit ca eticheta in interfata si live).
     */
    public String pseudonym;

    /**
     * Clasificatorul SVM antrenat pentru aceasta persoana (one‑vs‑all).
     */
    public SMO classifier;

    /**
     * Lista vectorilor HOG folositi la antrenamentul acestei persoane.
     * Utila pentru documentare, export si eventuale reantrenari.
     */
    public List<double[]> trainingHOGs;

    /**
     * Constructorul principal al modelului unei persoane.
     *
     * @param pseudonym    numele persoanei
     * @param classifier   clasificatorul SVM antrenat
     * @param trainingHOGs lista vectorilor HOG de antrenament
     */
    public PersonModel(String pseudonym, SMO classifier, List<double[]> trainingHOGs) {
        this.pseudonym = pseudonym;
        this.classifier = classifier;
        this.trainingHOGs = trainingHOGs;
    }
}