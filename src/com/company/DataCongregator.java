package com.company; /**
 * Created by jason on 3/5/17.
 */
import org.apache.commons.io.FileUtils;
import java.io.*;
import java.util.*;
import com.opencsv.*;


public class DataCongregator {

    public static ArrayList<String> getFeaturePaths(){
        File root = new File("/home/jason/Documents/MLProject/Image Data/benchmark");
        String fileName = "features.txt";
        ArrayList<String> Filepaths = new ArrayList<String>();
        try {
            boolean recursive = true;

            Collection files = FileUtils.listFiles(root, null, recursive);

            for (Iterator iterator = files.iterator(); iterator.hasNext();) {
                File file = (File) iterator.next();
                if (file.getName().equals(fileName))
                    Filepaths.add(file.getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return Filepaths;

    }
    public static ArrayList<String> setDataToList(String filepath){
        ArrayList<String> data = new ArrayList<String>();
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))){

            String sCurrentLine;
            while ((sCurrentLine = br.readLine()) != null){
                data.add(sCurrentLine);

            }


        }
        catch (IOException e){
            e.printStackTrace();
        }
        return data;

    }
    public static String[] splitData(ArrayList<String> dataHold){
        String[] CSV = (dataHold.get(0).split("[\\W]+"));
        return CSV;
    }

    public static ArrayList<Double> setDataToDouble(String[] CSV){
        ArrayList<Double> intData = new ArrayList<Double>();
        for(int i = 0; i < CSV.length; i++){
            intData.add(Double.parseDouble(CSV[i]));


        }
        return intData;

    }
    public static void writetoCSV(String[] CSV) throws IOException{
        FileWriter writer = new FileWriter("/home/jason/Documents/MLProject/Image_Data.csv", true);
        for (int j = 0; j < CSV.length; j++){
            writer.append(CSV[j]);
            writer.append(',');
        }
        writer.append("\n");
        writer.close();

    }




    public static void main(String[] args) throws IOException{
        ArrayList<String> Filepaths = new ArrayList<String>();
        ArrayList<String> dataHold = new ArrayList<String>();
        DataCongregator dataCong = new DataCongregator();
        Filepaths = dataCong.getFeaturePaths();
        System.out.println(Arrays.toString(Filepaths.toArray())+"\n");
        for(int i = 0; i < Filepaths.size(); i++){
            dataHold = dataCong.setDataToList((Filepaths.get(i)));
            for(int j = 0; j < dataHold.size(); j++){
                String[] CSV = (dataHold.get(j).split("[\t ]+"));
                dataCong.writetoCSV(CSV);
            }
        }




    }
}

