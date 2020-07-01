package main

import (
	//"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	LogisticRegression "github.com/Yushgoel/LogisticRegressionGo/logisticmodel"
)

func main() {
	// Open the file
	var x_train [][]float64
	var x_test [][]float64

	var y_train []float64
	var y_test []float64

	csvfile, err := os.Open("data.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)
	//r := csv.NewReader(bufio.NewReader(csvfile))
	counter := 0
	// Iterate through the records
	for {
		// Read each record from csv
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		if counter == 0 {
			fmt.Printf("Question: %s Answer %s\n", record[0], record[1])
		} else {
			var row []float64
			for i := 1; i < 2; i++ {
				record, _ := strconv.ParseFloat(record[i], 8)
				row = append(row, record)
			}
			y, _ := strconv.ParseFloat(record[0], 8)
			if counter < 712 {
				x_train = append(x_train, row)
				y_train = append(y_train, y)
			} else {
				x_test = append(x_test, row)
				y_test = append(y_test, y)
			}
		}
		counter++
	}
	fmt.Println(len(x_train))
	LogisticRegression.Model(x_train, y_train, x_test, y_test, 500, 0.05)
}
