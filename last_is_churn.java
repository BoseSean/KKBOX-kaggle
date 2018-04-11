import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.Scanner;

public class last_is_churn{

	static int lineCounter = 0;
	static float churn_rate = 0;
	static int churn_count = 0;
	static int last_1_is_churn = 0;
	static int last_2_is_churn = 0;
	static int last_3_is_churn = 0;
	static int last_4_is_churn = 0;
	static int last_5_is_churn = 0;
	static String currentMsno = "";
	static String[] nextMsno = null;
	
	public static void main(String[] args){
		String result = processImportTxt("C:/ZhangYuance/JavaHome/Workspace/Data_Process");
		System.out.println(result);
	}

	public static String processImportTxt(String sourceDirectory) {
			

			FileInputStream inputStream = null;
			FileOutputStream out = null;
			Scanner sc = null;
			File outputFile = null;;

	
			try {
				File rootDir = new File(sourceDirectory);
				
				for (File csv : rootDir.listFiles()) {
	
					if (csv.getAbsolutePath().endsWith("for_last.csv")) {
						
					    try {
					    	System.out.println("\n=======================================================================");
					    	System.out.println("Start to process " + csv + "\n");
							inputStream = new FileInputStream(csv);
							BufferedWriter bop = null;
							sc = new Scanner(inputStream, "UTF-8");
							
							sc.nextLine();
							//skip first line
							
							while (sc.hasNext()) {
								
								outputFile = new File(sourceDirectory + "/last_is_churn.csv");
								System.out.println("New File creadted: " + outputFile + "\n");
								
								if (!outputFile.exists()) {
									outputFile.createNewFile();
								}
								
								bop = new BufferedWriter(new FileWriter(outputFile));
								setExportFileContentandHeaders(bop);
								
								int persent = 0;
								int total = 0;
								while (sc.hasNext()) {
									sc = ReadBlock(sc);
									bop = writeLine(bop);
									
									//specially deal with the last line
									if (!sc.hasNext()) {
										sc = ReadBlock(sc);
										bop = writeLine(bop);
									}
									
									//timer
									persent = lineCounter/230000;
									if (persent == 1) {
										total++;
										System.out.println("Progress: " + total + "%");
										lineCounter = 0;
									}
								}
								
								if (sc.ioException() != null) {
									throw sc.ioException();
								}
								
								System.out.println("Finish writing to " + outputFile + "\n");
								
								bop.flush();
								bop.close();
								
							}
							
						} catch (Exception e) {
							e.printStackTrace();
							
						} finally {
							
						    if (inputStream != null) {
						        inputStream.close();
						    }
						    if (sc != null) {
						        sc.close();
						    }
						}
					    
					    System.out.println("Finish processing::" + csv + "\n");
					}
			    
				}
				
			} catch (Exception e) {
				e.printStackTrace();
			} 
	
			return "Successfully process csv";
		}
	
	private static BufferedWriter writeLine(BufferedWriter bop) {

		try {
			bop.write(currentMsno);
			bop.write(",");
			bop.write(Integer.toString(last_1_is_churn));
			bop.write(",");
			bop.write(Integer.toString(last_2_is_churn));
			bop.write(",");
			bop.write(Integer.toString(last_3_is_churn));
			bop.write(",");
			bop.write(Integer.toString(last_4_is_churn));
			bop.write(",");
			bop.write(Integer.toString(last_5_is_churn));
			bop.write(",");
			
			//change type of churn_rate to string
			String churnRateString = Float.toString(churn_rate);
			bop.write(churnRateString);
			bop.write(",");
			bop.write(Integer.toString(churn_count));
			
			bop.write(System.getProperty("line.separator"));
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return bop;
	}

	public static void setExportFileContentandHeaders(BufferedWriter output) throws IOException {
		
//'msno', 'last_1_is_churn',
  //      'last_2_is_churn', 'last_3_is_churn', 'last_4_is_churn', 'last_5_is_churn', 'churn_rate',
   //     'churn_count'
		output.write("msno");
		output.write(",");
		
		output.write("last_1_is_churn");
		output.write(",");
		
		output.write("last_2_is_churn");
		output.write(",");
		
		output.write("last_3_is_churn");
		output.write(",");
		
		output.write("last_4_is_churn");
		output.write(",");
		
		output.write("last_5_is_churn");
		output.write(",");
		
		output.write("churn_rate");
		output.write(",");
		
		output.write("churn_count");
		
		output.write(System.getProperty("line.separator"));

		return;
	}
	
	//check if the the transaction record is churn
	private static int isChurn(ArrayList<String[]> lines, int pos) {
		
		String expireDateString = lines.get(pos)[2];
		LocalDate expireDate = LocalDate.parse(expireDateString, DateTimeFormatter.BASIC_ISO_DATE);
		LocalDate EndTime = LocalDate.parse("20170331",DateTimeFormatter.BASIC_ISO_DATE);
		
		//if expireDate is after 20170331 considered as not churn
		if (expireDate.isAfter(EndTime)) 
			return 0;
		
		String nextTransactionDateString = lines.get(pos+1)[1];
		LocalDate nextTransaction = LocalDate.parse(nextTransactionDateString, DateTimeFormatter.BASIC_ISO_DATE);
		
		//if nextTransaction date is after or equal to 30days after the expiration date, it is considered as churn
		LocalDate checkDate = expireDate.plusDays(30);
		if(nextTransaction.isAfter(checkDate) || nextTransaction.isEqual(checkDate))
			return 1;
		
		return 0;
	}
	
	//check whether the last transaction record is churn
	private static int lastIsChurn(String expireDateString) {
		
		LocalDate expireDate = LocalDate.parse(expireDateString, DateTimeFormatter.BASIC_ISO_DATE);
		LocalDate EndTime = LocalDate.parse("20170401",DateTimeFormatter.BASIC_ISO_DATE);
		//if expireDate is after 20170331 considered as not churn
		if (expireDate.isAfter(EndTime)) 
			return 0;

		LocalDate checkDate = expireDate.plusDays(30);
		//if 30days after last transaction is before 20170401, considered as churn
		if(checkDate.isBefore(EndTime))
			return 1;
		
		return 0;
	}
	private static Scanner ReadBlock(Scanner sc) {
		
		//initialization
		churn_rate = 0;
		churn_count = 0;
		last_1_is_churn = 0;
		last_2_is_churn = 0;
		last_3_is_churn = 0;
		last_4_is_churn = 0;
		last_5_is_churn = 0;	
		
		ArrayList<String[]> lines = new ArrayList<String[]>();
		int counter = 1;
		
		String line = null;
		String[] msno = null;
		
		
		if (nextMsno != null) {
			msno = nextMsno;
		}
		else {
			//the first line of the data set
			line = sc.nextLine();
			msno = line.split(",");
		}
		
		lines.add(msno);
		currentMsno = msno[0];
		
		//if havent reached the end of file 
		if (sc.hasNext()) {
			String nextLine = sc.nextLine();
			String[] nextmsno = nextLine.split(",");
			nextMsno = nextmsno;
			
			//Read the whole block
			while(msno[0].equalsIgnoreCase(nextmsno[0])) {
				counter++;
				lines.add(nextmsno);
				msno = nextmsno;
				if (sc.hasNext()) {
					line = sc.nextLine();
					nextmsno = line.split(",");
					nextMsno = nextmsno;
				}
				else break;
			}
			
			//process the whole block
			if (counter <= 5) {
				switch(counter) {
				case 2:
						//last_is_churn_2
						last_2_is_churn = isChurn(lines,0);
						break;
				case 3:
						//last_is_churn_3
						last_3_is_churn = isChurn(lines,0);
						last_2_is_churn = isChurn(lines,1);
						break;
				case 4:
						//last_is_churn_4
						last_4_is_churn = isChurn(lines,0);
						last_3_is_churn = isChurn(lines,1);
						last_2_is_churn = isChurn(lines,2);
						break;
				case 5:
						//last_is_churn_5
						last_5_is_churn = isChurn(lines,0);
						last_4_is_churn = isChurn(lines,1);
						last_3_is_churn = isChurn(lines,2);
						last_2_is_churn = isChurn(lines,3);
						break;
				}
				
			}
			else {
				int length = lines.size();
				
				for(int i=0; i<length-5; i++)
					if (isChurn(lines,i) == 1)
						churn_count++;
				
				last_5_is_churn = isChurn(lines,length - 5);
				last_4_is_churn = isChurn(lines,length - 4);
				last_3_is_churn = isChurn(lines,length - 3);
				last_2_is_churn = isChurn(lines,length - 2);
			}
			
			//do last is churn
			last_1_is_churn = lastIsChurn(lines.get(lines.size()-1)[2]);
			churn_count += last_1_is_churn + last_2_is_churn + last_3_is_churn + last_4_is_churn + last_5_is_churn;
			
			churn_rate = (float)churn_count/counter;
			lineCounter += counter;
		}
		else {
			//if the whole block is only one line and reach the end of file
			//do last is churn
			last_1_is_churn = lastIsChurn(lines.get(lines.size()-1)[2]);
			
			if (last_1_is_churn == 1)
				churn_count++;
			churn_rate = churn_count/1;
		}

		return sc;
	}
}