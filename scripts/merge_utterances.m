%%%
% Author: Elton Martinez
% Last Modifier: 9/13/2023
% This function checks the difference between the offset of the current 
% utterance and the onset of the next utterance, for all pairs in the
% transcription file. If this difference is less than the given gap value
% the utterances are merged
%
% Input: (path, gap_between_utterance,outpath)  
% Output: A txt file containing the merged utterances based on the
% gap_between_utterance value(in seconds), if the write input is 1 is then it will
% be saved as a txt file otherwise one can save it as a variable(cell)
%%%
 
function merged_transcription = merge_utterances(path, gap_between_utterance,write)

    % Read in the transcription file and format it into a table 
    t = transcription_to_table(path);
    merged = cell(0,3);

    % Outerloop ensures that the loop does not run for more than the total
    % amount of instances
    i = 1;
    while i ~= height(t)
        final_utterance = "";
        j = i;
        
        % Will check the difference between the ith instance and the i+1
        % instance is less than the set threshold, if so then merge the
        % utterance and check if the next utterance is valid. Do so until 
        % merging criteria is not meet 
        while true & (j ~= height(t))
            diff = t{j+1,"onset"}{1} - t{j,"offset"}{1};

            % if the difference of the gap is greater than or equal to
            % gap_between_utterance then there's no merge exit second loop
            if diff >= gap_between_utterance

                % if i is not equal to j then that means that more than one
                % transcription instance was merged, add new values to cell
                if i ~= j
                    merged{end+1,1} = t{i,"onset"}{1};
                    merged{end,2} = t{j,"offset"}{1};
                    merged{end,3} = final_utterance;
                    % Since we merged j-i instaces we can skip to j as our
                    % new starting point
                    i = j;
                end
                break
            end
            
            % Merge the two instances 
            str1 = t{j,"utterance"}{1};
            str2 = t{j+1,"utterance"}{1};
            combined = append(str1," ",str2);
            % Merge the merged instace to the previous merged instance
            final_utterance = append(final_utterance," ",combined);
            
            % move on to the next merge candidate instance
            j = j + 1;
        end
        % Move on to the next transcription instance
        i = i + 1;
    end
    
    merged_transcription = merged;

     % if write is one then return a cell otherwise return a csv
     if write == 1
         writecell(merged,"merged_transcription.txt","Delimiter","\t", ...
                                            "QuoteStrings","none")
     end