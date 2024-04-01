// App.js
import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import WelcomeScreen from '../StudentModelApp/pages/WelcomeScreen';
import AppScreen from '../StudentModelApp/pages/AppScreen';

const Stack = createStackNavigator();

function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false, // This will hide the header globally for all screens
        }}
        initialRouteName="Welcome"
      >
        <Stack.Screen name="Welcome" component={WelcomeScreen} />
        <Stack.Screen name="MainApp" component={AppScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default App;
