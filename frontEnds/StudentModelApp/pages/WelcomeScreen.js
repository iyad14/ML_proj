// WelcomeScreen.js
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Image } from 'react-native';

const WelcomeScreen = ({ navigation }) => (
    <View style={styles.container}>
        <Image source={require('../images/logo.png')} style={styles.logo} />
        <Text style={styles.welcomeText}>Let's predict your future</Text>
        <TouchableOpacity onPress={() => navigation.navigate('MainApp')} style={styles.beginButton}>
            <Text style={styles.beginButtonText}>Begin</Text>
        </TouchableOpacity>
    </View>
);

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#121212', // Full black background
    },
    logo: {
        width: 300, // Larger logo width
        height: 300, // Larger logo height
        resizeMode: 'contain', // Ensure full logo is visible and aspect ratio is maintained
    },
    welcomeText: {
        fontSize: 24,
        color: '#FFF',
        marginVertical: 20,
    },
    beginButton: {
        paddingHorizontal: 20,
        paddingVertical: 10,
        backgroundColor: '#841584',
        borderRadius: 5,
    },
    beginButtonText: {
        color: '#FFF',
        fontSize: 20,
    },
});

// Make sure to export WelcomeScreen
export default WelcomeScreen;
