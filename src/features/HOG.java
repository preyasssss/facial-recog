package features;

import java.awt.image.BufferedImage;

/**
 * Implementare proprie a descriptorului HOG (Histogram of Oriented Gradients).
 * <p>
 * Algoritmul:
 * <ol>
 *   <li>Converteste imaginea la tonuri de gri (daca nu este deja).</li>
 *   <li>Calculeaza gradientii orizontali si verticali folosind filtrul [-1,0,1],
 *       apoi magnitudinea si orientarea fiecarui pixel (orientare intre 0 si 180 grade).</li>
 *   <li>Imparte imaginea in celule de 8x8 pixeli si acumuleaza o histograma cu 9 bin-uri
 *       (cate 20 de grade fiecare), folosind votare biliniara intre bin-urile adiacente.</li>
 *   <li>Grupeaza celulele in blocuri de 2x2 celule (suprapuse la fiecare 8 pixeli),
 *       normalizeaza fiecare bloc cu norma L2 si concateneaza toate blocurile
 *       intr-un vector descriptor final.</li>
 * </ol>
 * <p>
 * Acest descriptor este folosit atat in detectorul de capete (ferestre 64x64) cat si
 * in recunoasterea faciala (imagini 128x128).
 */
public class HOG {

    /** Dimensiunea unei celule in pixeli (8x8). */
    public static final int CELL_SIZE = 8;

    /** Numarul de celule pe directia OX si OY intr-un bloc (2x2). */
    public static final int BLOCK_CELLS = 2;

    /** Numarul de bin-uri din histograma de orientari (0-180 grade). */
    public static final int BINS = 9;

    /**
     * Extrage descriptorul HOG dintr-o imagine data.
     * <p>
     * Daca imaginea nu este deja in tonuri de gri, o converteste automat.
     *
     * @param input imaginea sursa (color sau gri)
     * @return vectorul HOG normalizat; lungimea depinde de dimensiunea imaginii
     */
    public static double[] extract(BufferedImage input) {
        // Asiguram imagine in tonuri de gri
        BufferedImage gray;
        if (input.getType() != BufferedImage.TYPE_BYTE_GRAY) {
            gray = capture.ImageUtils.toGrayscale(input);
        } else {
            gray = input;
        }

        int width  = gray.getWidth();
        int height = gray.getHeight();

        // Matrici pentru magnitudine si orientare
        double[][] mag = new double[height][width];
        double[][] ang = new double[height][width];

        // Calcul gradient cu operatorul [-1, 0, 1]
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int gx = getGray(gray, x + 1, y) - getGray(gray, x - 1, y);
                int gy = getGray(gray, x, y + 1) - getGray(gray, x, y - 1);
                mag[y][x] = Math.sqrt(gx * gx + gy * gy);
                double angle = Math.toDegrees(Math.atan2(gy, gx));
                if (angle < 0) angle += 180;
                ang[y][x] = angle;
            }
        }

        int cellsX = width  / CELL_SIZE;
        int cellsY = height / CELL_SIZE;

        // Histograma 3D: [celulaY][celulaX][bin]
        double[][][] cellHist = new double[cellsY][cellsX][BINS];

        // Umplere histograme pe celule
        for (int cy = 0; cy < cellsY; cy++) {
            for (int cx = 0; cx < cellsX; cx++) {
                for (int dy = 0; dy < CELL_SIZE; dy++) {
                    for (int dx = 0; dx < CELL_SIZE; dx++) {
                        int iy = cy * CELL_SIZE + dy;
                        int ix = cx * CELL_SIZE + dx;
                        if (iy < height && ix < width) {
                            double m = mag[iy][ix];
                            double a = ang[iy][ix];

                            // Determinare bin principal
                            int bin = (int) (a / 20.0);
                            if (bin == BINS) bin = 0;

                            // Votare biliniara intre bin-ul principal si unul adiacent
                            double centre = bin * 20 + 10;
                            double diff   = (a - centre) / 20;
                            double vote   = m * (1 - Math.abs(diff));

                            cellHist[cy][cx][bin] += vote;

                            if (diff > 0 && bin < BINS - 1) {
                                cellHist[cy][cx][bin + 1] += m * Math.abs(diff);
                            } else if (diff < 0 && bin > 0) {
                                cellHist[cy][cx][bin - 1] += m * Math.abs(diff);
                            } else if (bin == 0 && diff < 0) {
                                cellHist[cy][cx][BINS - 1] += m * Math.abs(diff);
                            } else if (bin == BINS - 1 && diff > 0) {
                                cellHist[cy][cx][0] += m * Math.abs(diff);
                            }
                        }
                    }
                }
            }
        }

        int blockStride = CELL_SIZE;   // blocurile aluneca la fiecare 8 pixeli
        int blocksX = (width  - BLOCK_CELLS * CELL_SIZE) / blockStride + 1;
        int blocksY = (height - BLOCK_CELLS * CELL_SIZE) / blockStride + 1;
        int descSize = blocksY * blocksX * BLOCK_CELLS * BLOCK_CELLS * BINS;
        double[] descriptor = new double[descSize];
        int idx = 0;

        // Construire blocuri si normalizare L2
        for (int by = 0; by < blocksY; by++) {
            for (int bx = 0; bx < blocksX; bx++) {
                double[] blockVec = new double[BLOCK_CELLS * BLOCK_CELLS * BINS];
                int v = 0;
                for (int bcy = 0; bcy < BLOCK_CELLS; bcy++) {
                    for (int bcx = 0; bcx < BLOCK_CELLS; bcx++) {
                        int cellY = by * blockStride / CELL_SIZE + bcy;
                        int cellX = bx * blockStride / CELL_SIZE + bcx;
                        if (cellY < cellsY && cellX < cellsX) {
                            for (int k = 0; k < BINS; k++) {
                                blockVec[v++] = cellHist[cellY][cellX][k];
                            }
                        } else {
                            v += BINS;
                        }
                    }
                }
                // Normalizare L2 cu un epsilon pentru stabilitate
                double norm = 0.0;
                for (double val : blockVec) norm += val * val;
                norm = Math.sqrt(norm + 1e-6);
                for (int i = 0; i < blockVec.length; i++) {
                    descriptor[idx++] = blockVec[i] / norm;
                }
            }
        }
        return descriptor;
    }

    /**
     * Obtine valoarea de gri a unui pixel (media aritmetica R+G+B).
     */
    private static int getGray(BufferedImage img, int x, int y) {
        int rgb = img.getRGB(x, y);
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8)  & 0xFF;
        int b =  rgb        & 0xFF;
        return (r + g + b) / 3;
    }
}