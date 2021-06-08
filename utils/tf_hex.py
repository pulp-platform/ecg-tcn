#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
def c_array_maker(model_data, model_name):

    c_str = ''
    c_str += '/*----------------------------------------------------------------------------*/\n'
    c_str += '/* Copyright (C) 2021 ETH Zurich, Switzerland                                 */\n'
    c_str += '/* SPDX-License-Identifier: Apache-2.0                                        */\n'
    c_str += '/*                                                                            */\n'
    c_str += '/* Licensed under the Apache License, Version 2.0 (the "License");            */\n'
    c_str += '/* you may not use this file except in compliance with the License.           */\n'
    c_str += '/* You may obtain a copy of the License at                                    */\n'
    c_str += '/*                                                                            */\n'
    c_str += '/* http://www.apache.org/licenses/LICENSE-2.0                                 */\n'
    c_str += '/*                                                                            */\n'
    c_str += '/* Unless required by applicable law or agreed to in writing, software        */\n'
    c_str += '/* distributed under the License is distributed on an "AS IS" BASIS,          */\n'
    c_str += '/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   */\n'
    c_str += '/* See the License for the specific language governing permissions and        */\n'
    c_str += '/* limitations under the License.                                             */\n'
    c_str += '/*                                                                            */\n'
    c_str += '/* Author:  Thorir Mar Ingolfsson                                             */\n'
    c_str += '/*----------------------------------------------------------------------------*/\n\n\n'

    c_str += '#ifndef ' + model_name.upper() + '_H\n'
    c_str += '#define ' + model_name.upper() + '_H\n\n'
    c_str += '\nunsigned int ' + model_name + '_len = ' + str(len(model_data)) + ';\n'
    c_str += 'unsigned char ' + model_name + '[] = {'
    hex_array = []
    for i, val in enumerate(model_data) :
        hex_str = format(val, '#04x')
        if (i + 1) < len(model_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)
    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'
    c_str += '#endif //' + model_name.upper() + '_H'
    return c_str