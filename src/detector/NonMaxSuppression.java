package detector;

import java.util.*;

/**
 * Implementeaza algoritmul de suprimare non-maxima (Non‑Maximum Suppression, NMS)
 * utilizat pentru a reduce multiplele detecții suprapuse ale aceleiasi fete,
 * pastrand doar ferestrele cele mai bune (cu scor maxim).
 * <p>
 * Algoritmul sorteaza detecțiile descrescator dupa scor, apoi iterativ
 * selecteaza cea mai buna detecție si elimina toate celelalte care au
 * o suprapunere (IoU) peste un prag dat.
 */
public class NonMaxSuppression {

    /**
     * Suprima detecțiile redundante dintr‑o lista.
     *
     * @param detections   lista initiala de detecții (dreptunghiuri cu scor)
     * @param iouThreshold pragul IoU (Intersection over Union) peste care
     *                     doua detecții sunt considerate ca apartinand
     *                     aceluiasi obiect
     * @return o lista filtrata, fara suprapuneri majore
     */
    public static List<HeadDetector.Detection> suppress(
            List<HeadDetector.Detection> detections, double iouThreshold) {

        List<HeadDetector.Detection> sorted = new ArrayList<>(detections);
        // Sorteaza descrescator dupa scor
        sorted.sort((a, b) -> Double.compare(b.score, a.score));

        List<HeadDetector.Detection> picked = new ArrayList<>();
        while (!sorted.isEmpty()) {
            HeadDetector.Detection best = sorted.remove(0);
            picked.add(best);
            // Elimina detecțiile care se suprapun prea mult cu cea aleasa
            sorted.removeIf(d -> iou(best, d) > iouThreshold);
        }
        return picked;
    }

    /**
     * Calculeaza intersectia peste uniune (IoU) a doua dreptunghiuri.
     *
     * @param a primul dreptunghi
     * @param b al doilea dreptunghi
     * @return valoarea IoU, intre 0 (fara suprapunere) si 1 (identice)
     */
    private static double iou(HeadDetector.Detection a, HeadDetector.Detection b) {
        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.width, b.x + b.width);
        int y2 = Math.min(a.y + a.height, b.y + b.height);

        if (x2 <= x1 || y2 <= y1) return 0; // nu exista intersectie

        double inter = (x2 - x1) * (y2 - y1);
        double areaA = a.width * a.height;
        double areaB = b.width * b.height;
        return inter / (areaA + areaB - inter);
    }
}