
// AppScreen.js
import React, { useState } from 'react';
import {
    ScrollView,
    View,
    TextInput,
    Button,
    Text,
    TouchableOpacity,
    StyleSheet,
    SafeAreaView,
} from 'react-native';
import axios from 'axios';
import Icon from 'react-native-vector-icons/MaterialIcons';

Icon.loadFont();

const translatePrediction = (value) => {
    const categories = {
        0: 'Dropout',
        1: 'Enrolled',
        2: 'Graduate',
    };
    return categories[value] || 'Unknown';
};

const Checkbox = ({ name, isChecked, onPress }) => (
    <View style={styles.checkboxContainer}>
        <Text style={styles.checkboxLabel}>{name.replace(/_/g, ' ')}</Text>
        <TouchableOpacity
            onPress={onPress}
            style={[
                styles.checkbox,
                {
                    backgroundColor: isChecked === null ? '#333' : isChecked ? 'green' : 'red',
                },
            ]}
        >
            <Icon name={isChecked ? 'check' : 'close'} size={24} color="#FFF" />
        </TouchableOpacity>
    </View>
);

const ResultRow = ({ model, prediction, confidence }) => (
    <View style={styles.tableRowContainer}>
        <Text style={styles.tableCell}>{model}</Text>
        <Text style={styles.tableCell}>{prediction}</Text>
        <Text style={styles.tableCell}>{confidence}</Text>
    </View>
);

const AppScreen = () => {
    const [studentData, setStudentData] = useState({
        curricular_units_2nd_sem_approved: '',
        curricular_units_2nd_sem_grade: '',
        curricular_units_1st_sem_approved: '',
        curricular_units_1st_sem_grade: '',
        admission_grade: '',
        tuition_fees_up_to_date: null,
        scholarship_holder: null,
        curricular_units_2nd_sem_enrolled: '',
        curricular_units_1st_sem_enrolled: '',
        displaced: null,
    });

    const [results, setResults] = useState([]);

    const handleInputChange = (name, value) => {
        setStudentData({ ...studentData, [name]: value });
    };

    const handleCheckboxChange = (name) => {
        const currentValue = studentData[name];
        setStudentData({ ...studentData, [name]: currentValue === null ? true : currentValue === true ? false : null });
    };

    const handleSubmit = async () => {
        const payload = Object.keys(studentData).reduce((acc, key) => {
            acc[key] = typeof studentData[key] === 'boolean' ? (studentData[key] ? 1 : 0) : studentData[key];
            return acc;
        }, {});

        try {
            const response = await axios.post('http://172.20.10.12:8000/predict/', payload);
            const processedResults = [];
            let knnCount = 0;
            let rfCount = 0;

            Object.entries(response.data).forEach(([key, value]) => {
                if (key.includes('Prediction')) {
                    const isKNN = key.includes('KNN');
                    const model = isKNN ? (++knnCount > 1 ? 'KNN 2nd Prediction' : 'KNN') : (++rfCount > 1 ? 'RF 2nd Prediction' : 'RF');
                    const confidenceKey = key.replace('Prediction', 'Confidence');
                    const confidenceValue = response.data[confidenceKey] * 100;

                    processedResults.push({
                        model,
                        prediction: translatePrediction(value),
                        confidence: `${confidenceValue.toFixed(2)}%`,
                    });
                }
            });
            setResults(processedResults);
        } catch (error) {
            console.error(error);
            alert('Failed to make prediction.');
        }
    };

    return (
        <SafeAreaView style={styles.safeArea}>
            <ScrollView contentContainerStyle={styles.container}>
                <Text style={styles.title}>Student Data Input</Text>
                {Object.keys(studentData).filter(key => !['tuition_fees_up_to_date', 'scholarship_holder', 'displaced'].includes(key)).map((key) => (
                    <TextInput
                        key={key}
                        style={styles.input}
                        onChangeText={(value) => handleInputChange(key, value)}
                        placeholder={key.replace(/_/g, ' ')}
                        placeholderTextColor="#999"
                        keyboardType="numeric"
                        value={studentData[key]}
                    />
                ))}
                <Checkbox
                    name="Tuition fees up to date"
                    isChecked={studentData.tuition_fees_up_to_date}
                    onPress={() => handleCheckboxChange('tuition_fees_up_to_date')}
                />
                <Checkbox
                    name="Scholarship holder"
                    isChecked={studentData.scholarship_holder}
                    onPress={() => handleCheckboxChange('scholarship_holder')}
                />
                <Checkbox
                    name="Displaced"
                    isChecked={studentData.displaced}
                    onPress={() => handleCheckboxChange('displaced')}
                />
                <Button title="Submit" onPress={handleSubmit} color="#841584" />
                {results.length > 0 && (
                    <View style={styles.resultsTable}>
                        <View style={styles.tableHeaderRow}>
                            <Text style={styles.tableHeader}>Model</Text>
                            <Text style={styles.tableHeader}>Prediction</Text>
                            <Text style={styles.tableHeader}>Confidence</Text>
                        </View>
                        {results.map((result, index) => (
                            <ResultRow key={index} model={result.model} prediction={result.prediction} confidence={result.confidence} />
                        ))}
                    </View>
                )}
            </ScrollView>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    safeArea: {
        flex: 1,
        backgroundColor: '#121212',
    },
    container: {
        padding: 20,
    },
    input: {
        backgroundColor: '#333',
        color: '#FFF',
        borderWidth: 1,
        borderColor: '#555',
        borderRadius: 5,
        padding: 10,
        marginBottom: 10,
    },
    checkboxContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 20,
        padding: 10,
        backgroundColor: '#222',
        borderRadius: 5,
    },
    checkboxLabel: {
        flex: 1,
        color: '#FFF',
    },
    checkbox: {
        padding: 5,
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 2,
        borderRadius: 5,
        borderColor: '#777',
    },
    title: {
        fontSize: 20,
        fontWeight: 'bold',
        alignSelf: 'center',
        marginBottom: 20,
        color: '#FFF',
    },
    resultsTable: {
        marginTop: 20,
    },
    tableHeaderRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        backgroundColor: '#333',
        borderTopWidth: 1,
        borderTopColor: '#555',
    },
    tableRowContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        padding: 10,
        borderBottomWidth: 1,
        borderBottomColor: '#555',
    },
    tableHeader: {
        flex: 1,
        color: '#FFF',
        fontWeight: 'bold',
        textAlign: 'center',
        padding: 10,
    },
    tableCell: {
        flex: 1,
        color: '#FFF',
        textAlign: 'center',
    },
});

export default AppScreen;