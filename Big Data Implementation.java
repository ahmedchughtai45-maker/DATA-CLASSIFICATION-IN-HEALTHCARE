import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

public class PatientClassification {
    private static final String FIXED_DELIM = "|||";
    private static final int EXPECTED_FIELDS = 12;

    public static void main(String[] args) throws Exception {
        Logger.getRootLogger().setLevel(Level.ERROR);

        // Expect two command-line arguments: HDFS input file and HDFS output directory
        if (args.length < 2) {
            System.err.println("Usage: PatientClassification <input_hdfs_path> <output_hdfs_directory>");
            System.exit(1);
        }
        String hdfsInputFile = args[0];          // /user/hadoop/input/MedicalFiles.csv
        String hdfsOutputDir = args[1];          // /user/hadoop/output

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        // Verify that the input file exists on HDFS
        Path inputPathHDFS = new Path(hdfsInputFile);
        if (!fs.exists(inputPathHDFS)) {
            System.err.println("Input file does not exist in HDFS at: " + hdfsInputFile);
            System.exit(1);
        }

        Path outputDirPath = new Path(hdfsOutputDir);
        if (fs.exists(outputDirPath)) {
            fs.delete(outputDirPath, true);
        }
        fs.mkdirs(outputDirPath);

        // Define HDFS output filenames for the three CSV files
        String hdfsNHSOutputFile = hdfsOutputDir + (hdfsOutputDir.endsWith("/") ? "" : "/") + "NHSNumber_filtered.csv";
        String hdfsSymptomsOutputFile = hdfsOutputDir + (hdfsOutputDir.endsWith("/") ? "" : "/") + "Symptoms_filtered.csv";
        String hdfsRegionOutputFile = hdfsOutputDir + (hdfsOutputDir.endsWith("/") ? "" : "/") + "NHS_Trust_Region_filtered.csv";

        // Define the header order used for CSV outputs
        String[] headers = {
            "PatientNHSNumber", "Name", "Age", "Gender", "Date_of_Admission",
            "Medical_Record_Number", "Medical_History", "Chief_Complaint",
            "History_of_Present_Illness", "Physical_Examination",
            "Assessment_and_Plan", "NHS_Trust_Region"
        };

        // Write the three CSV outputs directly to HDFS using FSDataOutputStream and BufferedWriter
        FSDataOutputStream outNHSStream = fs.create(new Path(hdfsNHSOutputFile));
        FSDataOutputStream outSymptomsStream = fs.create(new Path(hdfsSymptomsOutputFile));
        FSDataOutputStream outRegionStream = fs.create(new Path(hdfsRegionOutputFile));
        BufferedWriter writerNHS = new BufferedWriter(new OutputStreamWriter(outNHSStream, "UTF-8"));
        BufferedWriter writerSymptoms = new BufferedWriter(new OutputStreamWriter(outSymptomsStream, "UTF-8"));
        BufferedWriter writerRegion = new BufferedWriter(new OutputStreamWriter(outRegionStream, "UTF-8"));

        // Write header lines into each output file
        writerNHS.write(String.join(",", headers));
        writerNHS.newLine();

        List<String> symptomsHeaderList = new ArrayList<>();
        symptomsHeaderList.add(headers[7]); // Chief_Complaint first
        for (int i = 0; i < headers.length; i++) {
            if (i != 7) {
                symptomsHeaderList.add(headers[i]);
            }
        }
        writerSymptoms.write(String.join(",", symptomsHeaderList));
        writerSymptoms.newLine();

        List<String> regionHeaderList = new ArrayList<>();
        regionHeaderList.add(headers[11]); // NHS_Trust_Region first
        for (int i = 0; i < headers.length; i++) {
            if (i != 11) {
                regionHeaderList.add(headers[i]);
            }
        }
        writerRegion.write(String.join(",", regionHeaderList));
        writerRegion.newLine();

        // Open and read the CSV file from HDFS
        FSDataInputStream fsInputStream = fs.open(inputPathHDFS);
        BufferedReader br = new BufferedReader(new InputStreamReader(fsInputStream, "UTF-8"));
        // Skip the header line from the input CSV file
        String headerLineCSV = br.readLine();
        String line;
        while ((line = br.readLine()) != null) {
            if (line.trim().isEmpty())
                continue;
            String[] rawFields = parseCSVLine(line);
            String[] fields = adjustFields(rawFields);
            if (fields.length != EXPECTED_FIELDS)
                continue;
            // Write record to NHSNumber CSV
            writerNHS.write(String.join(",", fields));
            writerNHS.newLine();

            // For Symptoms CSV
            List<String> symptomsRow = new ArrayList<>();
            symptomsRow.add(fields[7]);
            for (int i = 0; i < EXPECTED_FIELDS; i++) {
                if (i != 7) {
                    symptomsRow.add(fields[i]);
                }
            }
            writerSymptoms.write(String.join(",", symptomsRow));
            writerSymptoms.newLine();

            // For Region CSV
            List<String> regionRow = new ArrayList<>();
            regionRow.add(fields[11]);
            for (int i = 0; i < EXPECTED_FIELDS; i++) {
                if (i != 11) {
                    regionRow.add(fields[i]);
                }
            }
            writerRegion.write(String.join(",", regionRow));
            writerRegion.newLine();
        }
        writerNHS.close();
        writerSymptoms.close();
        writerRegion.close();
        br.close();

        // MapReduce Classification Query Part (interactive via console)
        // Use an MR output directory under the provided HDFS output directory
        String mrOutputDir = hdfsOutputDir + (hdfsOutputDir.endsWith("/") ? "" : "/") + "temp_output";

        Scanner scanner = new Scanner(System.in);
        boolean continueQueries = true;
        while (continueQueries) {
            System.out.println("\nSelect classification criteria:");
            System.out.println("1. PatientID");
            System.out.println("2. Symptom");
            System.out.println("3. Region");
            System.out.print("Enter your selection (1, 2, 3) or type 'exit' to quit: ");
            String option = scanner.nextLine().trim();
            if (option.equalsIgnoreCase("exit"))
                break;

            String userDefinedCriteria = "";
            String userSpecifiedValue = "";
            switch (option) {
                case "1":
                    userDefinedCriteria = "patientId";
                    System.out.print("Enter the PatientID: ");
                    userSpecifiedValue = scanner.nextLine().trim();
                    break;
                case "2":
                    userDefinedCriteria = "symptom";
                    System.out.print("Enter the Symptom: ");
                    userSpecifiedValue = scanner.nextLine().trim();
                    break;
                case "3":
                    userDefinedCriteria = "region";
                    System.out.print("Enter the Region: ");
                    userSpecifiedValue = scanner.nextLine().trim();
                    break;
                default:
                    System.err.println("Invalid selection.");
                    continue;
            }
            Configuration jobConf = new Configuration();
            jobConf.set("userDefinedCriteria", userDefinedCriteria);
            jobConf.set("userSpecifiedValue", userSpecifiedValue);
            Job job = Job.getInstance(jobConf, "Patient Record Classification Query");
            job.setJarByClass(PatientClassification.class);
            job.setMapperClass(TokenizerMapper.class);
            job.setReducerClass(ClassificationReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            // Use the same HDFS input file for the MapReduce job
            FileInputFormat.addInputPath(job, new Path(hdfsInputFile));
            Path mrOutputPath = new Path(mrOutputDir);
            FileOutputFormat.setOutputPath(job, mrOutputPath);
            if (fs.exists(mrOutputPath)) {
                fs.delete(mrOutputPath, true);
            }
            boolean jobCompleted = job.waitForCompletion(true);
            if (jobCompleted) {
                System.out.println("\n==== Query Result for " + userDefinedCriteria +
                        " = " + userSpecifiedValue + " ====");
                FileStatus[] statuses = fs.listStatus(mrOutputPath);
                boolean foundOutput = false;
                for (FileStatus status : statuses) {
                    if (status.getPath().getName().startsWith("part-")) {
                        foundOutput = true;
                        FSDataInputStream in = fs.open(status.getPath());
                        IOUtils.copyBytes(in, System.out, 4096, false);
                        in.close();
                    }
                }
                if (!foundOutput) {
                    System.out.println("No records found for the given query.");
                }
                fs.delete(mrOutputPath, true);
            } else {
                System.err.println("MapReduce job failed.");
            }
            System.out.print("\nDo you want to perform another query? (yes/no): ");
            String response = scanner.nextLine().trim();
            if (response.equalsIgnoreCase("no") || response.equalsIgnoreCase("n")) {
                continueQueries = false;
            }
        }
        scanner.close();
        System.exit(0);
    }

    // Mapper class with region detection functionality
    public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, Text> {
        private String userDefinedCriteria;
        private String userSpecifiedValue;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            userDefinedCriteria = conf.get("userDefinedCriteria");
            userSpecifiedValue = conf.get("userSpecifiedValue");
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty() || line.startsWith("PatientNHSNumber"))
                return;
            String[] rawFields = parseCSVLine(line);
            String[] fields = adjustFields(rawFields);
            if (fields.length != EXPECTED_FIELDS) {
                return;
            }
            String classificationKey = "";
            String fieldValue = "";
            switch (userDefinedCriteria) {
                case "patientId":
                    classificationKey = fields[0].trim();
                    if (fields[0].trim().equalsIgnoreCase(userSpecifiedValue.trim())) {
                        fieldValue = userSpecifiedValue.trim();
                    }
                    break;
                case "symptom":
                    classificationKey = fields[7].trim();
                    String chiefComplaint = fields[7].replaceAll("\"", "").toLowerCase();
                    String searchSymptom = userSpecifiedValue.trim().toLowerCase();
                    if (chiefComplaint.contains(searchSymptom)) {
                        fieldValue = userSpecifiedValue.trim();
                    }
                    break;
                case "region":
                    // region detection logic:
                    String regionFieldRaw = fields[EXPECTED_FIELDS - 1].trim().replaceAll("\"", "").trim();
                    String normalizedRegionField = regionFieldRaw.replace("_", " ").toLowerCase();
                    String normalizedUserValue = userSpecifiedValue.trim().replace("_", " ").toLowerCase();
                    classificationKey = regionFieldRaw;
                    if (normalizedRegionField.contains(normalizedUserValue)) {
                        fieldValue = userSpecifiedValue.trim();
                    }
                    break;
                default:
                    return;
            }
            if (!fieldValue.isEmpty()) {
                String adjustedLine = String.join(FIXED_DELIM, fields);
                context.write(new Text(classificationKey), new Text(adjustedLine));
            }
        }

        private String[] parseCSVLine(String line) {
            return line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
        }

        private String[] adjustFields(String[] rawFields) {
            String[] fields = new String[EXPECTED_FIELDS];
            for (int i = 0; i < 4; i++) {
                if (i < rawFields.length) {
                    fields[i] = rawFields[i];
                } else {
                    fields[i] = "";
                }
            }
            if (rawFields.length > 5 &&
                (rawFields[4].matches(".*-[A-Za-z]{3}") ||
                 rawFields[4].matches("[0-9]{1,2}-[A-Za-z]{3}")) &&
                rawFields[5].matches("[0-9]{4}")) {
                fields[4] = rawFields[4] + " " + rawFields[5];
                for (int i = 6; i < rawFields.length && (i - 1) < EXPECTED_FIELDS; i++) {
                    fields[i - 1] = rawFields[i];
                }
            } else {
                for (int i = 4; i < Math.min(rawFields.length, EXPECTED_FIELDS); i++) {
                    fields[i] = rawFields[i];
                }
            }
            for (int i = 0; i < EXPECTED_FIELDS; i++) {
                if (fields[i] == null) {
                    fields[i] = "";
                }
            }
            return fields;
        }
    }

    // Reducer class
    public static class ClassificationReducer extends Reducer<Text, Text, Text, Text> {
        private final String[] headers = {
            "PatientNHSNumber", "Name", "Age", "Gender", "Date_of_Admission",
            "Medical_Record_Number", "Medical_History", "Chief_Complaint",
            "History_of_Present_Illness", "Physical_Examination",
            "Assessment_and_Plan", "NHS_Trust_Region"
        };

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            Set<String> uniqueRecords = new LinkedHashSet<>();
            for (Text value : values) {
                uniqueRecords.add(value.toString());
            }
            for (String record : uniqueRecords) {
                String formatted = formatRecord(record);
                context.write(null, new Text(formatted));
            }
        }

        private String formatRecord(String record) {
            String[] fields = record.split("\\Q" + FIXED_DELIM + "\\E");
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < Math.min(headers.length, fields.length); i++) {
                sb.append(headers[i]).append(": ").append(fields[i].trim()).append("\n");
            }
            sb.append("------------------------------------------------------------\n");
            return sb.toString();
        }
    }

    // Helper functions for CSV processing (used in both main and the Mapper)
    private static String[] parseCSVLine(String line) {
        return line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
    }

    private static String[] adjustFields(String[] rawFields) {
        String[] fields = new String[EXPECTED_FIELDS];
        for (int i = 0; i < 4; i++) {
            if (i < rawFields.length) {
                fields[i] = rawFields[i];
            } else {
                fields[i] = "";
            }
        }
        if (rawFields.length > 5 &&
            (rawFields[4].matches(".*-[A-Za-z]{3}") || rawFields[4].matches("[0-9]{1,2}-[A-Za-z]{3}")) &&
            rawFields[5].matches("[0-9]{4}")) {
            fields[4] = rawFields[4] + " " + rawFields[5];
            for (int i = 6; i < rawFields.length && (i - 1) < EXPECTED_FIELDS; i++) {
                fields[i - 1] = rawFields[i];
            }
        } else {
            for (int i = 4; i < Math.min(rawFields.length, EXPECTED_FIELDS); i++) {
                fields[i] = rawFields[i];
            }
        }
        for (int i = 0; i < EXPECTED_FIELDS; i++) {
            if (fields[i] == null) {
                fields[i] = "";
            }
        }
        return fields;
    }
}
