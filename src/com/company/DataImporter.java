package com.company; /**
 * Created by jason on 3/5/17.
 */
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class DataImporter {
    private static final String FILENAME = "/home/jason/Documents/ML Project/Image Data/benchmark/saiapr_tc-12/00/features.txt";

    public static void main(String[] args) {

        BufferedReader br = null;
        FileReader fr = null;

        try {

            fr = new FileReader(FILENAME);
            br = new BufferedReader(fr);

            String sCurrentLine;

            br = new BufferedReader(new FileReader(FILENAME));

            while ((sCurrentLine = br.readLine()) != null) {
                System.out.println(sCurrentLine);
            }

        } catch (IOException e) {

            e.printStackTrace();

        } finally {

            try {

                if (br != null)
                    br.close();

                if (fr != null)
                    fr.close();

            } catch (IOException ex) {

                ex.printStackTrace();

            }

        }

    }

}

